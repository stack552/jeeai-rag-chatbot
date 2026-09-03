"""
check_full_text.py
---------------------
Checks your ChromaDB collection to see how many chunks are missing
the full_text field in metadata (meaning generate_answer() silently
falls back to using the shorter SEARCH_SUMMARY instead).
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_PATH = r"C:\Users\nsaip\OneDrive\Desktop\Kinematics\chroma_db"
COLLECTION_NAME = "kinematics_lectures"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

all_data = collection.get(include=["metadatas"])
total = len(all_data["ids"])

missing_ids = []
for chunk_id, meta in zip(all_data["ids"], all_data["metadatas"]):
    if "full_text" not in meta or not meta.get("full_text"):
        missing_ids.append(chunk_id)

missing = len(missing_ids)

print(f"{missing} out of {total} chunks are missing full_text\n")

if missing_ids:
    print("Missing chunk IDs:")
    for cid in missing_ids:
        print(f"  - {cid}")