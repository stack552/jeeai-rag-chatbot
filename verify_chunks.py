"""
Verify what's currently stored in your ChromaDB collection.
Run this anytime to double check chunk contents are correct.
"""

import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="kinematics_lectures",
    embedding_function=embedding_fn
)

result = collection.get()  # fetches all stored chunks

print(f"Total chunks in collection: {len(result['ids'])}\n")

for i, chunk_id in enumerate(result["ids"]):
    print("=" * 60)
    print(f"CHUNK ID: {chunk_id}")
    print(f"METADATA: {result['metadatas'][i]}")
    print(f"TEXT:\n{result['documents'][i]}")
    print()