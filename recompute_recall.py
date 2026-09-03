"""
recompute_recall.py
---------------------
eval_results.json was generated (with real LLM answers + retrieved_ids)
BEFORE ground_truth.json's chunk_id fields were fixed from hyphens to
underscores. This means expected_chunk_retrieved was computed against
the WRONG chunk_id format and is incorrect.

This script does NOT call the LLM again - it just re-checks each
result's already-saved retrieved_ids against the CORRECTED chunk_id
from ground_truth.json, and updates eval_results.json in place.
"""

import json

with open("ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth_set = json.load(f)

with open("eval_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# Build a lookup: question text -> correct chunk_id (from the FIXED ground_truth.json)
correct_chunk_ids = {item["question"]: item["chunk_id"] for item in ground_truth_set}

hits = 0
for r in results:
    correct_id = correct_chunk_ids.get(r["question"])
    if correct_id is None:
        print(f"[warning] no matching ground truth question found for: {r['question'][:60]}")
        continue

    r["expected_chunk_id"] = correct_id  # update to the corrected value
    r["expected_chunk_retrieved"] = correct_id in r["retrieved_ids"]

    if r["expected_chunk_retrieved"]:
        hits += 1
    else:
        print(f"[MISS] {r['question'][:60]}...")
        print(f"        expected: {correct_id}")
        print(f"        got:      {r['retrieved_ids']}")

with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nRecomputed context recall: {hits}/{len(results)} = {hits/len(results)*100:.1f}%")
print("eval_results.json updated with corrected expected_chunk_retrieved values.")