"""
export_chunks.py
-------------------
Regenerates a fresh, up-to-date text export of all chunks currently
in your ChromaDB database - same format as your original
chunks_export.txt, but reflecting the current state (e.g. after the
full_text backfill fix).

Saves to chunks_export_updated.txt in the current folder.
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_PATH = r"C:\Users\nsaip\OneDrive\Desktop\Kinematics\chroma_db"
COLLECTION_NAME = "kinematics_lectures"
OUTPUT_FILE = "chunks_export_updated.txt"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

all_data = collection.get(include=["documents", "metadatas"])
total = len(all_data["ids"])

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(f"TOTAL CHUNKS: {total}\n\n")
    f.write("=" * 80 + "\n\n")

    for chunk_id, document, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
        f.write(f"CHUNK ID: {chunk_id}\n")
        f.write(f"Lecture: {meta.get('lecture_number', 'N/A')}\n")
        f.write(f"Title: {meta.get('title', 'N/A')}\n")
        f.write(f"Video ID: {meta.get('video_id', 'N/A')}\n")
        f.write(f"URL: {meta.get('url', 'N/A')}\n")
        f.write(f"Topic: {meta.get('topic', 'N/A')}\n")
        f.write(f"Subtopic: {meta.get('subtopic', 'N/A')}\n")
        f.write(f"Source: {meta.get('source', 'N/A')}\n")
        f.write(f"Slides Covered: {meta.get('slides_covered', 'N/A')}\n")
        f.write(f"Chunk Type: {meta.get('chunk_type', 'N/A')}\n")
        f.write("-" * 80 + "\n")
        f.write("SEARCH_SUMMARY (embedded, used for retrieval):\n")
        f.write(document + "\n")
        f.write("-" * 80 + "\n")
        f.write("FULL_CHUNK_TEXT (sent to LLM, not embedded):\n")
        full_text = meta.get("full_text")
        if full_text:
            f.write(full_text + "\n")
        else:
            f.write("[MISSING full_text in metadata]\n")
        f.write("=" * 80 + "\n\n")

print(f"Exported {total} chunks to {OUTPUT_FILE}")