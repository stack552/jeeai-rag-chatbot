"""
semantic_cache.py
-------------------
Flat semantic cache for question -> answer pairs, backed by Redis
(with RediSearch vector support, via the redis-stack-server image).

If a new question is semantically close enough to a previously cached
question, returns the cached answer instantly - no LLM call needed.

SELF-CONTAINED: uses the same embedding model as rag_pipeline.py
(all-MiniLM-L6-v2) so cache similarity is computed the same way as
retrieval similarity. Does not import from rag_pipeline.py or
langgraph_clarify.py, to avoid circular imports - gets wired into
app.py alongside them.
"""

import numpy as np
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
from redis.commands.search.index_definition import IndexDefinition, IndexType
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
REDIS_HOST = "localhost"
REDIS_PORT = 6379
INDEX_NAME = "qa_cache_idx"
KEY_PREFIX = "qa_cache:"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension
SIMILARITY_THRESHOLD = 0.90  # cosine similarity - tune this based on real testing
# -----------------------------

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def _create_index_if_missing():
    """Creates the Redis vector index for the cache, if it doesn't exist yet."""
    try:
        r.ft(INDEX_NAME).info()
        # Index already exists
    except redis.exceptions.ResponseError:
        schema = (
            TextField("question"),
            TextField("answer"),
            VectorField(
                "embedding",
                "FLAT",  # brute-force for small cache sizes; can switch to HNSW later
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH)
        r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
        print(f"[semantic_cache] Created new Redis index: {INDEX_NAME}")


_create_index_if_missing()


def _embed(text: str) -> np.ndarray:
    return embedding_model.encode(text).astype(np.float32)


def check_cache(question: str):
    """
    Checks if a semantically similar question is already cached.
    Returns the cached answer (str) if found above the similarity
    threshold, else None.
    """
    query_vector = _embed(question)

    query = (
        Query(f"*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("question", "answer", "score")
        .dialect(2)
    )

    try:
        results = r.ft(INDEX_NAME).search(
            query, query_params={"vec": query_vector.tobytes()}
        )
    except redis.exceptions.ResponseError:
        # Index empty or not ready yet
        return None

    if not results.docs:
        return None

    top = results.docs[0]
    # COSINE distance in RediSearch: score is distance (0 = identical), NOT similarity
    distance = float(top.score)
    similarity = 1 - distance

    if similarity >= SIMILARITY_THRESHOLD:
        print(f"[semantic_cache] HIT (similarity={similarity:.3f}) for: {question!r}")
        return top.answer.decode() if isinstance(top.answer, bytes) else top.answer

    print(f"[semantic_cache] MISS (best similarity={similarity:.3f}) for: {question!r}")
    return None


def store_in_cache(question: str, answer: str):
    """Stores a new question -> answer pair in the cache."""
    import hashlib

    key = KEY_PREFIX + hashlib.sha256(question.encode()).hexdigest()
    vector = _embed(question)

    r.hset(
        key,
        mapping={
            "question": question,
            "answer": answer,
            "embedding": vector.tobytes(),
        },
    )


# ---------- Standalone test ----------
if __name__ == "__main__":
    # Store a sample entry
    store_in_cache(
        "Why tau_A = 2*l*eta / (v0*(eta^2 - 1))?",
        "This comes from the boat-river problem: tau_A is the sum of downstream and upstream travel times..."
    )

    # Test exact match
    print("\n--- Test 1: exact match ---")
    result = check_cache("Why tau_A = 2*l*eta / (v0*(eta^2 - 1))?")
    print("Result:", result)

    # Test reworded version
    print("\n--- Test 2: reworded ---")
    result = check_cache("Can you explain why tau_A equals 2*l*eta over v0 times eta squared minus 1?")
    print("Result:", result)

    # Test genuinely different question
    print("\n--- Test 3: different question ---")
    result = check_cache("What is the difference between average speed and average velocity?")
    print("Result:", result)