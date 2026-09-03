"""
Add ONE manually-created chunk to your ChromaDB collection.

Usage: fill in the CHUNK section below with your chunk's text + metadata,
then run this script. Repeat for each new chunk you create.

Run this from the same folder each time so it reuses the same local
ChromaDB database (stored in ./chroma_db).
"""

import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------
# CHUNK: fill this in for each new chunk you add
# ---------------------------------------------------------
SEARCH_SUMMARY = """
Practical Application: v² - u² = 2as
(Read as: v squared minus u squared, equals 2 a s)
A cart on a track is used to demonstrate the third equation of motion
using photogates to measure velocity at two points.
Given:
v = 3.0 m/s (final velocity, measured at Photogate 2)
u = 1.0 m/s (initial velocity, measured at Photogate 1)
s = 2.0 m (displacement between Point A and Photogate 2)
(Read as: v equals 3.0 meters per second; u equals 1.0 meters per
second; s equals 2.0 meters)
Using the equation rearranged to solve for acceleration:
a = (v² - u²) / (2s)
(Read as: a equals, v squared minus u squared, divided by, 2 s)
a = (3² - 1²) / (2 × 2.0)
(Read as: a equals, 3 squared minus 1 squared, divided by, 2 times
2.0)
a = (9 - 1) / 4.0
(Read as: a equals, 9 minus 1, by 4.0)
a = 8 / 4.0
(Read as: a equals 8 by 4.0)
a = 2.0 m/s²
(Read as: a equals 2.0 meters per second squared)
This confirms the acceleration of the cart is 2.0 m/s², demonstrating
how v² = u² + 2as can be used experimentally to determine acceleration
from measured velocities and displacement, without needing to measure
time directly.
(Read as: v squared equals u squared, plus 2 a s)
""".strip()

CHUNK_TEXT = """
Practical Application: v² - u² = 2as
(Read as: v squared minus u squared, equals 2 a s)
A cart on a track is used to demonstrate the third equation of motion
using photogates to measure velocity at two points.
Given:
v = 3.0 m/s (final velocity, measured at Photogate 2)
u = 1.0 m/s (initial velocity, measured at Photogate 1)
s = 2.0 m (displacement between Point A and Photogate 2)
(Read as: v equals 3.0 meters per second; u equals 1.0 meters per
second; s equals 2.0 meters)
Using the equation rearranged to solve for acceleration:
a = (v² - u²) / (2s)
(Read as: a equals, v squared minus u squared, divided by, 2 s)
a = (3² - 1²) / (2 × 2.0)
(Read as: a equals, 3 squared minus 1 squared, divided by, 2 times
2.0)
a = (9 - 1) / 4.0
(Read as: a equals, 9 minus 1, by 4.0)
a = 8 / 4.0
(Read as: a equals 8 by 4.0)
a = 2.0 m/s²
(Read as: a equals 2.0 meters per second squared)
This confirms the acceleration of the cart is 2.0 m/s², demonstrating
how v² = u² + 2as can be used experimentally to determine acceleration
from measured velocities and displacement, without needing to measure
time directly.
(Read as: v squared equals u squared, plus 2 a s)
""".strip()

METADATA = {
  "lecture_number": 35,
  "video_id": "aX-6IK2whns",
  "title": "Kinematics Lecture 35 : proof of v² - u² = 2as || Ekalavya",
  "url": "https://www.youtube.com/watch?v=aX-6IK2whns",
  "topic": "Kinematics",
  "subtopic": "v² = u² + 2as - Practical application with photogates",
  "source": "Ekalavya ConceptLibrary",
  "slides_covered": [4],
  "chunk_type": "solved_example",
  "full_text": CHUNK_TEXT
}

CHUNK_ID = "lecture35_chunk2"

# ---------------------------------------------------------
# Below this line: no need to edit, just run
# ---------------------------------------------------------

def main():
    # Persistent local ChromaDB - saves to disk in ./chroma_db
    client = chromadb.PersistentClient(path="./chroma_db")

    # Use a good free local embedding model (sentence-transformers)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="kinematics_lectures",
        embedding_function=embedding_fn
    )

    # Add (or update if same ID already exists) this chunk
    collection.upsert(
        ids=[CHUNK_ID],
        documents=[SEARCH_SUMMARY],
        metadatas=[METADATA]
    )

    print(f"Added/updated chunk: {CHUNK_ID}")
    print(f"Collection now has {collection.count()} total chunks.")


if __name__ == "__main__":
    main()