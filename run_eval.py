"""
Retrieval Eval Script for Kinematics RAG Chatbot
--------------------------------------------------
Loads eval_questions.json, queries ChromaDB for each question,
and checks whether the expected CHUNK_ID(s) appear in the top-k results.

BEFORE RUNNING:
1. Update CHROMA_DB_PATH below if your path is different.
2. Update COLLECTION_NAME to match whatever you used when you created
   your collection (e.g. client.get_collection("your_name_here")).
3. Make sure sentence-transformers is installed:
   pip install sentence-transformers --break-system-packages
   (only needed if you embed manually instead of using ChromaDB's
   built-in embedding function)
"""

import json
import chromadb
from chromadb.utils import embedding_functions

# ---------- CONFIG: EDIT THESE ----------
# Use the SAME path you use in add_chunk.py. If you run add_chunk.py with
# path="./chroma_db" (relative), run this script from that same folder too,
# OR replace this with the full absolute path to that chroma_db folder.
CHROMA_DB_PATH = r"C:\Users\nsaip\OneDrive\Desktop\Kinematics\chroma_db"
COLLECTION_NAME = "kinematics_lectures"  # matches add_chunk.py
TOP_K = 5
EVAL_FILE = "eval_questions.json"
RESULTS_FILE = "eval_results.json"
# -----------------------------------------

# Must match add_chunk.py exactly, so queries are embedded the same way
# your chunks were embedded at insert time.
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def load_eval_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    eval_set = load_eval_questions(EVAL_FILE)
    results_log = []

    for item in eval_set:
        question = item["question"]
        expected = set(item["expected_chunk_ids"])

        # ChromaDB embeds the query text internally using the same
        # all-MiniLM-L6-v2 embedding_function attached to the collection.
        results = collection.query(query_texts=[question], n_results=TOP_K)

        retrieved_ids = results["ids"][0]
        retrieved_set = set(retrieved_ids)

        hit_ids = expected.intersection(retrieved_set)
        full_hit = expected.issubset(retrieved_set)
        partial_hit = len(hit_ids) > 0 and not full_hit
        miss = len(hit_ids) == 0

        # rank of first expected chunk found (1-indexed), None if not found
        rank = None
        for idx, cid in enumerate(retrieved_ids):
            if cid in expected:
                rank = idx + 1
                break

        status = "FULL_HIT" if full_hit else ("PARTIAL_HIT" if partial_hit else "MISS")

        results_log.append({
            "id": item["id"],
            "question": question,
            "type": item.get("type", ""),
            "expected_chunk_ids": list(expected),
            "retrieved_ids": retrieved_ids,
            "status": status,
            "rank_of_first_hit": rank,
        })

        print(f"[{item['id']}] {status:12s} | rank={rank} | Q: {question}")

    # ---- Summary metrics ----
    total = len(results_log)
    full_hits = sum(1 for r in results_log if r["status"] == "FULL_HIT")
    partial_hits = sum(1 for r in results_log if r["status"] == "PARTIAL_HIT")
    misses = sum(1 for r in results_log if r["status"] == "MISS")

    print("\n" + "=" * 50)
    print(f"TOTAL QUESTIONS : {total}")
    print(f"FULL HITS       : {full_hits} ({full_hits/total:.1%})")
    print(f"PARTIAL HITS    : {partial_hits} ({partial_hits/total:.1%})")
    print(f"MISSES          : {misses} ({misses/total:.1%})")
    print(f"Recall@{TOP_K} (full+partial): {(full_hits+partial_hits)/total:.1%}")
    print("=" * 50)

    # ---- Breakdown by question type ----
    by_type = {}
    for r in results_log:
        t = r["type"] or "unknown"
        by_type.setdefault(t, [0, 0])
        by_type[t][1] += 1
        if r["status"] == "FULL_HIT":
            by_type[t][0] += 1

    print("\nBreakdown by type (full hits / total):")
    for t, (h, tot) in by_type.items():
        print(f"  {t:10s}: {h}/{tot} = {h/tot:.1%}")

    # ---- Save results ----
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nFull results saved to {RESULTS_FILE}")

    # ---- Print MISSES for easy review ----
    misses_list = [r for r in results_log if r["status"] == "MISS"]
    if misses_list:
        print("\n" + "-" * 50)
        print("MISSES to review:")
        for r in misses_list:
            print(f"  [{r['id']}] Q: {r['question']}")
            print(f"       expected: {r['expected_chunk_ids']}")
            print(f"       got     : {r['retrieved_ids']}")


if __name__ == "__main__":
    run_eval()