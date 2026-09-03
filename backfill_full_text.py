"""
backfill_full_text.py
-------------------------
One-time fix: for every chunk missing full_text in metadata, copies
its existing SEARCH_SUMMARY (document) text into full_text. This is
safe because you've confirmed (by checking lecture1_chunk1 against
your original source) that SEARCH_SUMMARY already contains your
complete content for these concept-type chunks - it just never got
copied into full_text due to an earlier version of add_chunk.py that
didn't set that field.

This does NOT touch chunks that already have full_text set (e.g.
your "problem" type worked examples) - those are left untouched.
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

all_data = collection.get(include=["documents", "metadatas"])

ids_to_update = []
documents_to_keep = []
metadatas_to_update = []

for chunk_id, document, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
    if "full_text" not in meta or not meta.get("full_text"):
        # Backfill: copy the document (SEARCH_SUMMARY) into full_text
        updated_meta = dict(meta)  # copy, don't mutate original
        updated_meta["full_text"] = document
        ids_to_update.append(chunk_id)
        documents_to_keep.append(document)
        metadatas_to_update.append(updated_meta)

print(f"Found {len(ids_to_update)} chunks to backfill.")

if ids_to_update:
    confirm = input(f"Proceed with backfilling full_text for these {len(ids_to_update)} chunks? (yes/no): ")
    if confirm.strip().lower() == "yes":
        collection.upsert(
            ids=ids_to_update,
            documents=documents_to_keep,
            metadatas=metadatas_to_update,
        )
        print(f"Done. Backfilled full_text for {len(ids_to_update)} chunks.")
    else:
        print("Cancelled - no changes made.")
else:
    print("Nothing to backfill - all chunks already have full_text.")