INPUT_FILE = "chunks_export.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

blocks = content.split("=" * 80)

for block in blocks:
    if "Chunk Type: solved_example" in block and "[MISSING full_text in metadata]" in block:
        print(block.strip())
        print("\n" + "#" * 60 + "\n")