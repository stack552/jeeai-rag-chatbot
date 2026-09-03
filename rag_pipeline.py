"""
rag_pipeline.py
-----------------
Core RAG pipeline logic: takes a raw student question, rewrites it if
needed (to handle formula-heavy queries), retrieves matching chunks
from ChromaDB using hybrid retrieval (original + rewritten query),
maintains a running LRU pool of relevant chunks across a conversation,
and generates a grounded answer using Groq (with chat history for
conversational continuity). Answers are STREAMED word-by-word.

BEFORE RUNNING:
Set your Groq key as an environment variable (never hardcoded here -
this file is safe to keep in a public GitHub repo as-is):
    set GROQ_API_KEY=your_actual_key_here      (Windows CMD, per session)
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from langchain_groq import ChatGroq

# ---------- CONFIG ----------
CHROMA_DB_PATH = r"C:\Users\nsaip\OneDrive\Desktop\Kinematics\chroma_db"
COLLECTION_NAME = "kinematics_lectures"

# Reads your Groq key from an environment variable - never hardcoded here,
# so this file is safe to commit and push to a public GitHub repo.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable not set. Run this first: "
        "set GROQ_API_KEY=your_actual_key_here"
    )

TOP_K = 3
MAX_POOL_SIZE = 3
MAX_HISTORY_MESSAGES = 4

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

groq_client = Groq(api_key=GROQ_API_KEY)
langchain_llm = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY, temperature=0.3)


def rewrite_query(raw_question):
    prompt = f"""Rewrite this physics student's question about kinematics.
If it contains mathematical formulas, symbols, or notation, explain what
physical quantities and kinematics concept the formula relates to (e.g.
uniformly accelerated motion, velocity-displacement relation, projectile
motion, relative velocity) rather than just describing the math shape.
Keep it concise (1-2 sentences) and specific to kinematics vocabulary.

Student question: {raw_question}

Rewritten question:"""
    response = langchain_llm.invoke(prompt)
    return response.content.strip()


def handle_student_doubt(student_input, chat_history=None, turn_number=0, retrieved_chunks_pool=None):
    if chat_history is None:
        chat_history = []
    if retrieved_chunks_pool is None:
        retrieved_chunks_pool = {}
    turn_number += 1

    rewritten = rewrite_query(student_input)

    results_original = collection.query(query_texts=[student_input], n_results=TOP_K, include=["documents", "metadatas"])
    results_rewritten = collection.query(query_texts=[rewritten], n_results=TOP_K, include=["documents", "metadatas"])

    combined = {}
    for res in [results_original, results_rewritten]:
        for i, chunk_id in enumerate(res["ids"][0]):
            if chunk_id not in combined:
                combined[chunk_id] = {"document": res["documents"][0][i], "metadata": res["metadatas"][0][i]}

    final_ids = list(combined.keys())[:TOP_K]
    retrieved_ids = final_ids
    retrieved_summaries = [combined[cid]["document"] for cid in final_ids]
    retrieved_metadatas = [combined[cid]["metadata"] for cid in final_ids]
    retrieved_full_texts = [meta.get("full_text", summary) for meta, summary in zip(retrieved_metadatas, retrieved_summaries)]

    for chunk_id, full_text, meta in zip(retrieved_ids, retrieved_full_texts, retrieved_metadatas):
        retrieved_chunks_pool[chunk_id] = {"full_text": full_text, "metadata": meta, "last_touched_turn": turn_number}

    if len(retrieved_chunks_pool) > MAX_POOL_SIZE:
        sorted_ids = sorted(retrieved_chunks_pool.keys(), key=lambda cid: retrieved_chunks_pool[cid]["last_touched_turn"])
        num_to_remove = len(retrieved_chunks_pool) - MAX_POOL_SIZE
        for cid in sorted_ids[:num_to_remove]:
            del retrieved_chunks_pool[cid]

    pool_full_texts = [v["full_text"] for v in retrieved_chunks_pool.values()]
    pool_metadatas = [v["metadata"] for v in retrieved_chunks_pool.values()]

    answer = generate_answer(student_input, pool_full_texts, pool_metadatas, chat_history)

    chat_history.append({"role": "user", "content": student_input})
    chat_history.append({"role": "assistant", "content": answer})

    return {
        "original_question": student_input, "rewritten_question": rewritten,
        "retrieved_ids": retrieved_ids, "retrieved_summaries": retrieved_summaries,
        "retrieved_full_texts": retrieved_full_texts, "answer": answer,
        "chat_history": chat_history, "turn_number": turn_number,
        "retrieved_chunks_pool": retrieved_chunks_pool,
    }


def generate_answer(student_question, retrieved_full_texts, retrieved_metadatas, chat_history=None, top_n=None):
    if chat_history is None:
        chat_history = []
    chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
    context_chunks = retrieved_full_texts[:top_n] if top_n else retrieved_full_texts
    context_metas = retrieved_metadatas[:top_n] if top_n else retrieved_metadatas

    context_block = ""
    for i, (text, meta) in enumerate(zip(context_chunks, context_metas)):
        lecture_num = meta.get("lecture_number", "unknown")
        source = meta.get("source", "")
        context_block += f"\n--- Source {i+1} (Lecture {lecture_num}, {source}) ---\n{text}\n"

    system_prompt = """You are a physics tutor helping a JEE student with kinematics doubts.

RULES:
- Answer ONLY using the information in the provided context below.
- If the context does not contain enough information to answer confidently,
  say so honestly (e.g. "I don't have enough information on this specific
  point") rather than guessing or making up an explanation.
- When you use a source, mention which lecture it came from.
- Explain clearly and step-by-step, as if teaching a student who is confused.
- Do not introduce formulas, numbers, or facts that are not in the context."""

    user_prompt = f"""Context from lecture materials:
{context_block}

Student's question: {student_question}

Answer the student's question using only the context above."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    approx_input_chars = len(system_prompt) + len(user_prompt) + sum(len(m["content"]) for m in chat_history)
    approx_input_tokens = approx_input_chars // 4
    print(f"[token check] approx input tokens: {approx_input_tokens} (context window limit: 131,072)")

    stream = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b", messages=messages, max_tokens=2000, temperature=0.5, stream=True,
    )

    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_answer += delta
    print()

    return full_answer.strip()


# ============================================================
# ASYNC VERSIONS — for FastAPI backend (jeeai-backend/main.py).
# ============================================================

import asyncio
from groq import AsyncGroq

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)


async def rewrite_query_async(raw_question):
    prompt = f"""Rewrite this physics student's question about kinematics.
If it contains mathematical formulas, symbols, or notation, explain what
physical quantities and kinematics concept the formula relates to (e.g.
uniformly accelerated motion, velocity-displacement relation, projectile
motion, relative velocity) rather than just describing the math shape.
Keep it concise (1-2 sentences) and specific to kinematics vocabulary.

Student question: {raw_question}

Rewritten question:"""
    response = await langchain_llm.ainvoke(prompt)
    return response.content.strip()


async def generate_answer_stream_async(student_question, retrieved_full_texts, retrieved_metadatas, chat_history=None, top_n=None):
    if chat_history is None:
        chat_history = []
    chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
    context_chunks = retrieved_full_texts[:top_n] if top_n else retrieved_full_texts
    context_metas = retrieved_metadatas[:top_n] if top_n else retrieved_metadatas

    context_block = ""
    for i, (text, meta) in enumerate(zip(context_chunks, context_metas)):
        lecture_num = meta.get("lecture_number", "unknown")
        source = meta.get("source", "")
        context_block += f"\n--- Source {i+1} (Lecture {lecture_num}, {source}) ---\n{text}\n"

    system_prompt = """You are a physics tutor helping a JEE student with kinematics doubts.

RULES:
- Answer ONLY using the information in the provided context below.
- If the context does not contain enough information to answer confidently,
  say so honestly (e.g. "I don't have enough information on this specific
  point") rather than guessing or making up an explanation.
- When you use a source, mention which lecture it came from.
- Explain clearly and step-by-step, as if teaching a student who is confused.
- Do not introduce formulas, numbers, or facts that are not in the context."""

    user_prompt = f"""Context from lecture materials:
{context_block}

Student's question: {student_question}

Answer the student's question using only the context above."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    stream = await async_groq_client.chat.completions.create(
        model="openai/gpt-oss-120b", messages=messages, max_tokens=2000, temperature=0.5, stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def generate_answer_async(student_question, retrieved_full_texts, retrieved_metadatas, chat_history=None, top_n=None):
    full_answer = ""
    async for delta in generate_answer_stream_async(student_question, retrieved_full_texts, retrieved_metadatas, chat_history, top_n):
        full_answer += delta
    return full_answer.strip()


async def handle_student_doubt_async(student_input, chat_history=None, turn_number=0, retrieved_chunks_pool=None):
    if chat_history is None:
        chat_history = []
    if retrieved_chunks_pool is None:
        retrieved_chunks_pool = {}
    turn_number += 1

    rewritten = await rewrite_query_async(student_input)

    results_original = await asyncio.to_thread(collection.query, query_texts=[student_input], n_results=TOP_K, include=["documents", "metadatas"])
    results_rewritten = await asyncio.to_thread(collection.query, query_texts=[rewritten], n_results=TOP_K, include=["documents", "metadatas"])

    combined = {}
    for res in [results_original, results_rewritten]:
        for i, chunk_id in enumerate(res["ids"][0]):
            if chunk_id not in combined:
                combined[chunk_id] = {"document": res["documents"][0][i], "metadata": res["metadatas"][0][i]}

    final_ids = list(combined.keys())[:TOP_K]
    retrieved_ids = final_ids
    retrieved_summaries = [combined[cid]["document"] for cid in final_ids]
    retrieved_metadatas = [combined[cid]["metadata"] for cid in final_ids]
    retrieved_full_texts = [meta.get("full_text", summary) for meta, summary in zip(retrieved_metadatas, retrieved_summaries)]

    for chunk_id, full_text, meta in zip(retrieved_ids, retrieved_full_texts, retrieved_metadatas):
        retrieved_chunks_pool[chunk_id] = {"full_text": full_text, "metadata": meta, "last_touched_turn": turn_number}

    if len(retrieved_chunks_pool) > MAX_POOL_SIZE:
        sorted_ids = sorted(retrieved_chunks_pool.keys(), key=lambda cid: retrieved_chunks_pool[cid]["last_touched_turn"])
        num_to_remove = len(retrieved_chunks_pool) - MAX_POOL_SIZE
        for cid in sorted_ids[:num_to_remove]:
            del retrieved_chunks_pool[cid]

    pool_full_texts = [v["full_text"] for v in retrieved_chunks_pool.values()]
    pool_metadatas = [v["metadata"] for v in retrieved_chunks_pool.values()]

    answer = await generate_answer_async(student_input, pool_full_texts, pool_metadatas, chat_history)

    chat_history.append({"role": "user", "content": student_input})
    chat_history.append({"role": "assistant", "content": answer})

    return {
        "original_question": student_input, "rewritten_question": rewritten,
        "retrieved_ids": retrieved_ids, "retrieved_summaries": retrieved_summaries,
        "retrieved_full_texts": retrieved_full_texts, "answer": answer,
        "chat_history": chat_history, "turn_number": turn_number,
        "retrieved_chunks_pool": retrieved_chunks_pool,
    }


async def handle_student_doubt_stream_async(student_input, chat_history=None, turn_number=0, retrieved_chunks_pool=None):
    if chat_history is None:
        chat_history = []
    if retrieved_chunks_pool is None:
        retrieved_chunks_pool = {}
    turn_number += 1

    rewritten = await rewrite_query_async(student_input)

    results_original = await asyncio.to_thread(collection.query, query_texts=[student_input], n_results=TOP_K, include=["documents", "metadatas"])
    results_rewritten = await asyncio.to_thread(collection.query, query_texts=[rewritten], n_results=TOP_K, include=["documents", "metadatas"])

    combined = {}
    for res in [results_original, results_rewritten]:
        for i, chunk_id in enumerate(res["ids"][0]):
            if chunk_id not in combined:
                combined[chunk_id] = {"document": res["documents"][0][i], "metadata": res["metadatas"][0][i]}

    final_ids = list(combined.keys())[:TOP_K]
    retrieved_ids = final_ids
    retrieved_summaries = [combined[cid]["document"] for cid in final_ids]
    retrieved_metadatas = [combined[cid]["metadata"] for cid in final_ids]
    retrieved_full_texts = [meta.get("full_text", summary) for meta, summary in zip(retrieved_metadatas, retrieved_summaries)]

    for chunk_id, full_text, meta in zip(retrieved_ids, retrieved_full_texts, retrieved_metadatas):
        retrieved_chunks_pool[chunk_id] = {"full_text": full_text, "metadata": meta, "last_touched_turn": turn_number}

    if len(retrieved_chunks_pool) > MAX_POOL_SIZE:
        sorted_ids = sorted(retrieved_chunks_pool.keys(), key=lambda cid: retrieved_chunks_pool[cid]["last_touched_turn"])
        num_to_remove = len(retrieved_chunks_pool) - MAX_POOL_SIZE
        for cid in sorted_ids[:num_to_remove]:
            del retrieved_chunks_pool[cid]

    pool_full_texts = [v["full_text"] for v in retrieved_chunks_pool.values()]
    pool_metadatas = [v["metadata"] for v in retrieved_chunks_pool.values()]

    full_answer = ""
    async for delta in generate_answer_stream_async(student_input, pool_full_texts, pool_metadatas, chat_history):
        full_answer += delta
        yield {"type": "delta", "text": delta}

    chat_history.append({"role": "user", "content": student_input})
    chat_history.append({"role": "assistant", "content": full_answer})

    yield {
        "type": "final",
        "result": {
            "original_question": student_input, "rewritten_question": rewritten,
            "retrieved_ids": retrieved_ids, "retrieved_summaries": retrieved_summaries,
            "retrieved_full_texts": retrieved_full_texts, "answer": full_answer,
            "chat_history": chat_history, "turn_number": turn_number,
            "retrieved_chunks_pool": retrieved_chunks_pool,
        },
    }


if __name__ == "__main__":
    chat_history = None
    turn_number = 0
    pool = None

    print("Kinematics doubt-solver — type 'quit' to exit\n")

    while True:
        student_input = input("You: ").strip()
        if student_input.lower() in ("quit", "exit"):
            break

        result = handle_student_doubt(student_input, chat_history, turn_number, pool)

        print(f"\n[retrieved this turn: {result['retrieved_ids']}]")
        print(f"[pool now: {list(result['retrieved_chunks_pool'].keys())}]\n")

        chat_history = result["chat_history"]
        turn_number = result["turn_number"]
        pool = result["retrieved_chunks_pool"]