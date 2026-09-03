"""
dag_cache.py
-------------
DAG-based conversation cache: each node is one question+answer, linked
to its parent (the question that led to it) and children (follow-up
questions asked from it). Shared globally across all students - if a
new student's conversation path matches an existing path in the DAG,
they get the cached answer and reuse that path.

Uses RediSearch's HNSW vector index (same technique as semantic_cache.py)
for both local (scoped to a parent's children, via a TAG filter) and
global (whole-tree) similarity search - O(log n) instead of a manual
O(n) Python loop over every node.

SELF-CONTAINED: own Redis connection, own embedding model. No import
from rag_pipeline.py, langgraph_clarify.py, or semantic_cache.py, to
avoid circular imports. Gets wired into app.py separately.
"""

import json
import uuid
import numpy as np
import redis
from redis.commands.search.field import VectorField, TagField, TextField
from redis.commands.search.query import Query
from redis.commands.search.index_definition import IndexDefinition, IndexType
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
REDIS_HOST = "localhost"
REDIS_PORT = 6379
NODE_KEY_PREFIX = "dag_node:"
INDEX_NAME = "dag_idx"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension
ROOT_NODE_ID = "root"  # every conversation starts here
# -----------------------------

# Raw bytes (not decoded) - needed so the binary embedding field survives
# round-tripping through Redis untouched; text fields are decoded manually.
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def _embed(text: str) -> np.ndarray:
    return embedding_model.encode(text).astype(np.float32)


def _escape_tag(value: str) -> str:
    """RediSearch TAG queries treat hyphens as special characters -
    UUIDs are full of them, so they need escaping to be searched literally."""
    return value.replace("-", "\\-")


def _node_key(node_id: str) -> str:
    return f"{NODE_KEY_PREFIX}{node_id}"


def _create_index_if_missing():
    """Creates the RediSearch HNSW index over dag_node hashes, if it
    doesn't exist yet. TAG field on parent_id enables the "local" scoped
    search (find_matching_child); the VECTOR field enables both that
    and the "global" whole-tree search."""
    try:
        r.ft(INDEX_NAME).info()
    except redis.exceptions.ResponseError:
        schema = (
            TextField("question"),
            TextField("answer"),
            TagField("parent_id"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(prefix=[NODE_KEY_PREFIX], index_type=IndexType.HASH)
        r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
        print(f"[dag_cache] Created RediSearch HNSW index: {INDEX_NAME}")


_create_index_if_missing()


def create_node(question: str, answer: str, parent_id: str) -> str:
    """
    Creates a new node in the DAG for this question+answer, attached
    as a child of parent_id. Returns the new node's ID.
    """
    node_id = str(uuid.uuid4())
    embedding_bytes = _embed(question).tobytes()

    r.hset(
        _node_key(node_id),
        mapping={
            "question": question,
            "answer": answer,
            "parent_id": parent_id,
            "children_ids": json.dumps([]),
            "embedding": embedding_bytes,
        },
    )

    add_child_edge(parent_id, node_id)
    return node_id


def get_node(node_id: str) -> dict | None:
    """Returns the node's data as a dict, or None if it doesn't exist."""
    raw = r.hgetall(_node_key(node_id))
    if not raw:
        return None

    data = {k.decode(): v for k, v in raw.items()}
    embedding_bytes = data.pop("embedding", None)

    return {
        "question": data.get("question", b"").decode(),
        "answer": data.get("answer", b"").decode(),
        "embedding": np.frombuffer(embedding_bytes, dtype=np.float32) if embedding_bytes else None,
        "parent_id": data.get("parent_id", b"").decode(),
        "children_ids": json.loads(data["children_ids"].decode()) if data.get("children_ids") else [],
    }


def add_child_edge(parent_id: str, child_id: str):
    """Registers child_id as a child of parent_id."""
    parent_key = _node_key(parent_id)
    existing = r.hget(parent_key, "children_ids")
    children = json.loads(existing.decode()) if existing else []
    if child_id not in children:
        children.append(child_id)
    r.hset(parent_key, "children_ids", json.dumps(children))


def get_children(node_id: str) -> list:
    """Returns a list of child node dicts (each with 'id' included) for node_id."""
    node = get_node(node_id)
    if node is None:
        return []
    children = []
    for child_id in node["children_ids"]:
        child_data = get_node(child_id)
        if child_data:
            child_data["id"] = child_id
            children.append(child_data)
    return children


def ensure_root_exists():
    """Makes sure the ROOT_NODE_ID exists, so every conversation has a
    valid starting point to attach children to. Deliberately has NO
    embedding field - this naturally excludes it from KNN vector
    search results, since RediSearch can't match a vector query
    against a document missing that field."""
    if not r.exists(_node_key(ROOT_NODE_ID)):
        r.hset(
            _node_key(ROOT_NODE_ID),
            mapping={
                "question": "__ROOT__",
                "answer": "__ROOT__",
                "parent_id": "",
                "children_ids": json.dumps([]),
            },
        )
        print(f"[dag_cache] Created root node: {ROOT_NODE_ID}")


ensure_root_exists()


def find_matching_child(current_node_id: str, new_question: str, similarity_threshold: float = 0.90):
    """
    Checks if new_question semantically matches any EXISTING CHILD of
    current_node_id, using RediSearch's HNSW index with a TAG filter
    on parent_id (only search within this node's own children) -
    O(log n) instead of a manual Python loop.

    Returns the matching child node dict (with 'id') if found above
    the threshold, else None.
    """
    query_vector = _embed(new_question).tobytes()
    escaped_parent = _escape_tag(current_node_id)

    query = (
        Query(f"(@parent_id:{{{escaped_parent}}})=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("question", "answer", "score")
        .dialect(2)
    )

    try:
        results = r.ft(INDEX_NAME).search(query, query_params={"vec": query_vector})
    except redis.exceptions.ResponseError:
        return None

    if not results.docs:
        print("[dag_cache] No contextual match (no children indexed)")
        return None

    top = results.docs[0]
    distance = float(top.score)
    similarity = 1 - distance
    node_id = top.id[len(NODE_KEY_PREFIX):]

    if similarity >= similarity_threshold:
        print(f"[dag_cache] Contextual MATCH (similarity={similarity:.3f}): {top.question!r}")
        return {"id": node_id, "question": top.question, "answer": top.answer}

    print(f"[dag_cache] No contextual match (best similarity={similarity:.3f})")
    return None


def find_matching_node_globally(new_question: str, similarity_threshold: float = 0.90):
    """
    Searches the ENTIRE DAG (not just children of the current position)
    for a node whose stored question is semantically similar to
    new_question, using RediSearch's HNSW index - O(log n) instead of
    the old manual Python scan over every node. Used to avoid creating
    duplicate nodes when the same question is asked from a DIFFERENT
    position than where it was first asked, elsewhere in the tree.
    """
    query_vector = _embed(new_question).tobytes()

    query = (
        Query("*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("question", "answer", "score")
        .dialect(2)
    )

    try:
        results = r.ft(INDEX_NAME).search(query, query_params={"vec": query_vector})
    except redis.exceptions.ResponseError:
        return None

    if not results.docs:
        print("[dag_cache] No global match (index empty)")
        return None

    top = results.docs[0]
    distance = float(top.score)
    similarity = 1 - distance
    node_id = top.id[len(NODE_KEY_PREFIX):]

    if similarity >= similarity_threshold:
        print(f"[dag_cache] GLOBAL match found (similarity={similarity:.3f}): {top.question!r}")
        return {"id": node_id, "question": top.question, "answer": top.answer}

    print(f"[dag_cache] No global match (best similarity={similarity:.3f})")
    return None


# ---------- Standalone test ----------
if __name__ == "__main__":
    node1_id = create_node(
        "Why tau_A = 2*l*eta / (v0*(eta^2 - 1))?",
        "This comes from the boat-river problem, summing downstream and upstream times...",
        parent_id=ROOT_NODE_ID,
    )
    print(f"Created node1: {node1_id}")

    node2_id = create_node(
        "what about tau_B?",
        "tau_B comes from the across-river leg using the Pythagorean theorem for resultant velocity...",
        parent_id=node1_id,
    )
    print(f"Created node2 (child of node1): {node2_id}")

    print("\nRoot's children:")
    for child in get_children(ROOT_NODE_ID):
        print(f"  - {child['id']}: {child['question']!r}")

    print(f"\nNode1's children:")
    for child in get_children(node1_id):
        print(f"  - {child['id']}: {child['question']!r}")

    print("\n--- Contextual matching tests ---")
    match = find_matching_child(ROOT_NODE_ID, "Why tau_A = 2*l*eta / (v0*(eta^2 - 1))?")
    print(f"Exact repeat match found: {match is not None}")

    match = find_matching_child(node1_id, "can you also explain tau_B?")
    print(f"Reworded follow-up match found: {match is not None}")

    match = find_matching_child(node1_id, "What is the difference between speed and velocity?")
    print(f"Unrelated question match found: {match is not None}")