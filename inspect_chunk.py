"""
inspect_chunk.py
-------------------
Pulls one specific chunk's SEARCH_SUMMARY (document) text and prints
it along with its character/word count - so you can compare it
against your original source transcript/notes for that same lecture
to see if content is genuinely missing or just never copied into
full_text.
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_PATH = r"C:\Users\nsaip\OneDrive\Desktop\Kinematics\chroma_db"
COLLECTION_NAME = "kinematics_lectures"

# Change this to check a different chunk
CHUNK_ID_TO_CHECK = "lecture1_chunk1"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

result = collection.get(ids=[CHUNK_ID_TO_CHECK], include=["documents", "metadatas"])

if not result["ids"]:
    print(f"Chunk '{CHUNK_ID_TO_CHECK}' not found.")
else:
    document = result["documents"][0]
    metadata = result["metadatas"][0]

    print(f"CHUNK ID: {CHUNK_ID_TO_CHECK}")
    print(f"Character count: {len(document)}")
    print(f"Word count: {len(document.split())}")
    print(f"\nHas full_text in metadata: {'full_text' in metadata and bool(metadata.get('full_text'))}")
    print("\n" + "=" * 60)
    print("STORED SEARCH_SUMMARY / DOCUMENT TEXT:")
    print("=" * 60)
    print(document)