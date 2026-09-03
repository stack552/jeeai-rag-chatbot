"""
Scan chunks_export.txt and report which chunks are missing full_text,
broken down by chunk_type, so you know what's actually urgent to recover.

Usage: place this in the same folder as chunks_export.txt and run it.
"""

import re

INPUT_FILE = "chunks_export.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# Split into individual chunk blocks
blocks = content.split("=" * 80)

missing_by_type = {}
total_missing = 0
total_chunks = 0

for block in blocks:
    if "CHUNK ID:" not in block:
        continue
    total_chunks += 1

    chunk_id_match = re.search(r"CHUNK ID:\s*(.+)", block)
    lecture_match = re.search(r"Lecture:\s*(.+)", block)
    type_match = re.search(r"Chunk Type:\s*(.+)", block)

    chunk_id = chunk_id_match.group(1).strip() if chunk_id_match else "UNKNOWN"
    lecture = lecture_match.group(1).strip() if lecture_match else "?"
    chunk_type = type_match.group(1).strip() if type_match else "unknown"

    if "[MISSING full_text in metadata]" in block:
        total_missing += 1
        missing_by_type.setdefault(chunk_type, []).append((lecture, chunk_id))

print(f"Total chunks scanned: {total_chunks}")
print(f"Total missing full_text: {total_missing}\n")

print("Breakdown by chunk_type:")
for ctype, items in sorted(missing_by_type.items(), key=lambda x: -len(x[1])):
    print(f"  {ctype}: {len(items)} chunks missing")

print("\n" + "=" * 60)
print("HIGH PRIORITY chunks (problem / example type) missing full_text:")
print("=" * 60)
for ctype in missing_by_type:
    if ctype.lower() in ("problem", "example"):
        for lecture, chunk_id in sorted(missing_by_type[ctype], key=lambda x: str(x[0])):
            print(f"  Lecture {lecture}: {chunk_id}")

print("\n" + "=" * 60)
print("LOWER PRIORITY chunks (concept type) missing full_text:")
print("=" * 60)
for ctype in missing_by_type:
    if ctype.lower() == "concept":
        for lecture, chunk_id in sorted(missing_by_type[ctype], key=lambda x: str(x[0])):
            print(f"  Lecture {lecture}: {chunk_id}")