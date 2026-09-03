"""
run_ragas_eval_step2.py
-------------------------
STEP 2 of the RAGAS evaluation pipeline.

Reads ground_truth.json, runs EACH question through the real RAG
pipeline (rag_pipeline.handle_student_doubt - NOT the full app.py
DAG/clarification routing, since these are all self-contained,
specific questions meant to test the core RAG answer quality directly),
captures the actual generated answer + the retrieved context, and
saves everything to eval_results.json.

This does NOT run RAGAS itself yet - that's Step 3, which reads
eval_results.json.

UPDATE: added a deliberate pacing delay (PACING_DELAY_SECONDS) after
each successful question. Groq's llama-3.3-70b-versatile free tier
caps at 12,000 tokens/minute. Some "problem" type questions retrieve
large contexts (8000+ tokens), and combined with the pipeline's own
query-rewrite call, back-to-back questions can blow past that limit
within the same minute - causing repeated rate-limit failures even
after retries. Pacing keeps rolling token usage comfortably under
the per-minute cap instead of bursting and waiting out penalties.
"""

import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from groq import RateLimitError
from rag_pipeline import handle_student_doubt

INPUT_FILE = "ground_truth.json"
OUTPUT_FILE = "eval_results.json"

# Delay (seconds) after each successful question, to stay under the
# 12,000 tokens/minute free-tier cap on llama-3.3-70b-versatile.
# Increase this if you still hit rate limits; decrease if you have a
# paid tier with higher limits.
PACING_DELAY_SECONDS = 15


def load_existing_results():
    """Resume support: if eval_results.json already has some results,
    load them so we don't redo already-completed questions."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_one_with_retry(question, max_retries=3):
    """Runs handle_student_doubt, retrying on rate limit errors by
    waiting the time Groq tells us to wait."""
    for attempt in range(max_retries):
        try:
            return handle_student_doubt(question, chat_history=None, turn_number=0, retrieved_chunks_pool=None)
        except RateLimitError as e:
            wait_seconds = 60  # default fallback wait
            msg = str(e)
            # Try to parse "Please try again in Xm Ys" from the error message
            import re
            match = re.search(r"try again in (\d+)m([\d.]+)s", msg)
            if match:
                wait_seconds = int(match.group(1)) * 60 + float(match.group(2))
            print(f"  [rate limit hit] waiting {wait_seconds:.0f}s before retry {attempt+1}/{max_retries}...")
            time.sleep(wait_seconds + 5)  # small buffer
    raise RuntimeError(f"Failed after {max_retries} retries due to rate limits.")


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        ground_truth_set = json.load(f)

    results = load_existing_results()
    already_done_questions = {r["question"] for r in results}

    for i, item in enumerate(ground_truth_set):
        question = item["question"]

        if question in already_done_questions:
            print(f"[{i+1}/{len(ground_truth_set)}] Already done, skipping: {question[:60]}...")
            continue

        expected_chunk_id = item["chunk_id"]
        ground_truth = item["ground_truth"]

        print(f"\n[{i+1}/{len(ground_truth_set)}] Running: {question[:70]}...")

        result = run_one_with_retry(question)

        retrieved_ids = result["retrieved_ids"]
        expected_chunk_retrieved = expected_chunk_id in retrieved_ids

        results.append({
            "question": question,
            "answer": result["answer"],
            "contexts": result["retrieved_full_texts"],
            "ground_truth": ground_truth,
            "expected_chunk_id": expected_chunk_id,
            "retrieved_ids": retrieved_ids,
            "expected_chunk_retrieved": expected_chunk_retrieved,
        })

        print(f"  Expected chunk retrieved: {expected_chunk_retrieved}")

        # Save after EVERY question - so a crash never loses progress
        save_results(results)

        # Pace ourselves to stay under Groq's tokens-per-minute cap
        print(f"  Pausing {PACING_DELAY_SECONDS}s before next question...")
        time.sleep(PACING_DELAY_SECONDS)

    print(f"\n\nSaved {len(results)} results to {OUTPUT_FILE}")

    hits = sum(1 for r in results if r["expected_chunk_retrieved"])
    print(f"Context recall (expected chunk in top-5): {hits}/{len(results)} = {hits/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()