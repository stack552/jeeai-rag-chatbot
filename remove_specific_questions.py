"""
remove_specific_questions.py
-------------------------------
Removes specific questions (by exact text match) from
ragas_scores_progress.json, so the next run of run_ragas_eval_step3.py
will re-score them instead of skipping them as "already done".

Use this when a question got a suspicious/low score (e.g. context_recall
= 0.0 due to a judge-model scoring quirk) and you want a fresh attempt.
"""

import json

PROGRESS_FILE = "ragas_scores_progress.json"

# Edit this list - the exact question text (copy-paste from your CSV/JSON
# to avoid typos) for each question you want re-scored.
QUESTIONS_TO_RERUN = [
    "What are real world examples where we can say a body is at rest?",
    "What are real world examples where we can say an object is in motion?",
    "What is the difference between distance and displacement?",
    "What does speedometer of car show? Is it Average speed or Instantaneous speed?",
    "If a bullet is fired, the bullet goes up and comes down in projectile motion, while the monkey falls freely. How can I say at a specific angle 'theta' the bullet is not hitting the monkey at the moment when the bullet is fired? How can the motion of the bullet is not straight line sight?",
]
with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

before_count = len(results)
cleaned_results = [r for r in results if r.get("question") not in QUESTIONS_TO_RERUN]
after_count = len(cleaned_results)

removed = before_count - after_count

with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_results, f, indent=2, ensure_ascii=False)

print(f"Before: {before_count} entries")
print(f"After:  {after_count} entries")
print(f"Removed {removed} entries (out of {len(QUESTIONS_TO_RERUN)} requested).")

if removed != len(QUESTIONS_TO_RERUN):
    print("\n[warning] Not all requested questions were found/removed.")
    print("Double check the question text matches EXACTLY (including punctuation).")

print("\nRun run_ragas_eval_step3.py now - these questions will be re-scored,")
print("everything else will be skipped as before.")