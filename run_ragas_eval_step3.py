"""
run_ragas_eval_step3.py
-------------------------
STEP 3 of the RAGAS evaluation pipeline.

Reads eval_results.json (produced by run_ragas_eval_step2.py, and
corrected by recompute_recall.py) and runs RAGAS metrics on it:

  - faithfulness       : does the answer stick to the retrieved context,
                          or does it hallucinate / add unsupported claims?
  - answer_relevancy    : does the answer actually address the question?
  - context_precision   : of the chunks retrieved, how many were
                          actually relevant/needed?
  - context_recall      : did retrieval pull in what the ground_truth
                          answer needed?

UPDATE: now has RESUME SUPPORT, matching step2's design. Each question
is evaluated ONE AT A TIME (not as one big batch), and saved to
ragas_scores_progress.json immediately after. If you rerun this script
(e.g. after hitting a daily rate limit and coming back tomorrow),
questions already scored are SKIPPED - no wasted Groq calls, no
re-evaluating what's already done.

BEFORE RUNNING:
1. pip install ragas datasets langchain-groq langchain-huggingface (in ragas_env)
2. Set your Groq API key: set GROQ_API_KEY=your_key_here
3. Make sure eval_results.json exists in this folder.

NOTE ON EMBEDDINGS:
Uses a local HuggingFace embedding model instead of OpenAI's - no
extra API cost, runs on your machine.
"""

import json
import os

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.metrics import AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# Groq's API only supports n=1 generations per call, but RAGAS's default
# answer_relevancy metric requests n=3 (for self-consistency/robustness).
# strictness=1 forces it down to a single generation.
answer_relevancy = AnswerRelevancy(strictness=1)

INPUT_FILE = "eval_results.json"
PROGRESS_FILE = "ragas_scores_progress.json"  # resume-safe incremental save
OUTPUT_CSV = "ragas_scores.csv"

# ---------- CONFIG ----------
GROQ_MODEL = "openai/gpt-oss-120b"  # updated from deprecated llama-3.3-70b-versatile
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # matches your chunking setup
# -----------------------------


def load_eval_results():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_progress():
    """Resume support: load whatever's already been scored."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_progress(results):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY not set. Set it as an environment variable before running "
            "(see instructions at the top of this file)."
        )

    eval_results = load_eval_results()

    required_fields = ["question", "answer", "contexts", "ground_truth"]
    for i, r in enumerate(eval_results):
        missing = [f for f in required_fields if f not in r or r[f] in (None, "")]
        if missing:
            print(f"[warning] row {i} ('{r.get('question', '?')[:50]}') missing fields: {missing}")

    scored_results = load_progress()
    already_scored_questions = {r["question"] for r in scored_results}

    # Wire RAGAS's judge LLM to Groq
    judge_llm = LangchainLLMWrapper(
        ChatGroq(model=GROQ_MODEL, temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )

    # Keep concurrency low and timeout generous - free tier + heavy judge
    # calls means going fast just causes timeouts/rate limits.
    my_run_config = RunConfig(max_workers=2, timeout=120)

    print(f"Total questions: {len(eval_results)}. Already scored: {len(scored_results)}.\n")

    for i, row in enumerate(eval_results):
        question = row["question"]

        if question in already_scored_questions:
            print(f"[{i+1}/{len(eval_results)}] Already scored, skipping: {question[:60]}...")
            continue

        print(f"\n[{i+1}/{len(eval_results)}] Scoring: {question[:70]}...")

        # Build a single-row dataset for just this question
        single_row_data = {
            "question": [row["question"]],
            "answer": [row["answer"]],
            "contexts": [row["contexts"]],
            "ground_truth": [row["ground_truth"]],
        }
        dataset = Dataset.from_dict(single_row_data)

        try:
            score = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=judge_llm,
                embeddings=judge_embeddings,
                run_config=my_run_config,
            )
            score_df = score.to_pandas()
            score_row = score_df.iloc[0].to_dict()
            score_row["question"] = question

            # Check if ALL 4 metrics came back NaN (e.g. rate-limited mid-question).
            # If so, treat this as a failure, NOT a success - don't save it as
            # "done", so it gets retried on the next run instead of being
            # silently skipped forever with empty data.
            metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            import math
            all_nan = all(
                score_row.get(col) is None or (isinstance(score_row.get(col), float) and math.isnan(score_row.get(col)))
                for col in metric_cols
            )
            if all_nan:
                print(f"  [FAILED - all metrics NaN, likely rate limit] {question[:60]}...")
                print("  NOT saving as done - will retry on next run.")
                continue

            scored_results.append(score_row)
            print(f"  Done: {score_row}")

        except Exception as e:
            print(f"  [FAILED] {question[:60]}... - {type(e).__name__}: {e}")
            print("  Skipping for now - rerun the script later to retry just this one.")
            continue

        # Save after EVERY question - so a crash/rate-limit never loses progress
        save_progress(scored_results)

    print(f"\n\nScored {len(scored_results)}/{len(eval_results)} questions.")

    if scored_results:
        import pandas as pd
        final_df = pd.DataFrame(scored_results)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved full results to {OUTPUT_CSV}")

        print("\n" + "=" * 50)
        print("AVERAGE SCORES (across all scored questions so far)")
        print("=" * 50)
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in final_df.columns:
                print(f"{col}: {final_df[col].mean():.3f}")

    remaining = len(eval_results) - len(scored_results)
    if remaining > 0:
        print(f"\n{remaining} question(s) not yet scored (failed or rate-limited).")
        print("Just rerun this script later - already-scored questions will be skipped automatically.")


if __name__ == "__main__":
    main()