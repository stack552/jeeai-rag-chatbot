"""
cleanup_nan_entries.py
------------------------
Removes any entries from ragas_scores_progress.json where ALL 4 RAGAS
metrics are NaN (meaning that question was never actually scored -
it just got marked "done" by an earlier version of step3 that didn't
check for this). Run this once, then rerun run_ragas_eval_step3.py
to properly retry those questions.
"""

import json
import math

PROGRESS_FILE = "ragas_scores_progress.json"

with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def is_all_nan(row):
    for col in metric_cols:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            return False
    return True


before_count = len(results)
cleaned_results = [r for r in results if not is_all_nan(r)]
after_count = len(cleaned_results)

removed = before_count - after_count

with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_results, f, indent=2, ensure_ascii=False)

print(f"Before: {before_count} entries")
print(f"After:  {after_count} entries")
print(f"Removed {removed} all-NaN (failed) entries.")
if removed > 0:
    print("\nThese questions will be retried next time you run run_ragas_eval_step3.py")
else:
    print("\nNo bad entries found - progress file was already clean.")