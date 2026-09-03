"""
app.py
-------
Real entry point for the JEEAI kinematics doubt-solver. Wires together:
- rag_pipeline.py       (retrieval + generation, hybrid search, LRU pool,
                          REAL streaming via generate_answer())
- langgraph_clarify.py  (vague-question detection + clarifying questions)
- dag_cache.py          (shared DAG cache for self-contained specific questions)
- semantic_cache.py     (flat semantic cache - still used as a fast pre-check
                          once a conversation has gone vague)

RULE (finalized after design + testing):
- Once a conversation has ANY vague question, the DAG is permanently
  bypassed for the rest of that conversation - everything from then on
  is answered via handle_student_doubt() using chat_history + pool.
- Specific questions, in a conversation that hasn't gone vague yet, are
  checked against the DAG (scoped to the current node's children only).
  On a miss, they're answered via retrieval + generation WITHOUT
  chat_history (so the cached answer stays valid for any future student),
  then stored as a new DAG node.

DISPLAY:
- Answers that come from a FRESH LLM call (handle_student_doubt()) are
  already streamed live, word-by-word, inside rag_pipeline.generate_answer().
- Answers that come from a CACHE HIT (flat cache, DAG cache) or the
  clarifying question text already exist as complete strings - these are
  displayed with SIMULATED streaming (print_streamed) purely for a
  consistent look. This is NOT real generation streaming - it's a
  cosmetic word-by-word replay of an already-complete answer.
Each result dict includes "already_streamed": True/False so
run_interactive() knows which display method to use.

This is the file to actually run and to later wrap with FastAPI/Streamlit.
"""
import os
import time
import asyncio

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from rag_pipeline import handle_student_doubt, handle_student_doubt_async, handle_student_doubt_stream_async
from langgraph_clarify import (
    ConversationState,
    detect_vague_question,
    generate_clarifying_question,
    detect_vague_question_async,
    generate_clarifying_question_async,
    classify_question_type_async,
)
from semantic_cache import check_cache, store_in_cache
from dag_cache import find_matching_child, find_matching_node_globally, create_node, add_child_edge, ROOT_NODE_ID


def print_streamed(text, delay=0.02):
    """Prints a complete string word-by-word, simulating streaming, for
    answers that already exist in full (cache hits, clarifying questions).
    This is cosmetic only - no real generation is happening here."""
    words = text.split(" ")
    for i, word in enumerate(words):
        print(word, end=" " if i < len(words) - 1 else "", flush=True)
        time.sleep(delay)
    print()


def handle_student_doubt_with_clarification(
    student_input,
    chat_history=None,
    turn_number=0,
    retrieved_chunks_pool=None,
    pending_clarification=None,
    dag_current_node=ROOT_NODE_ID,
    conversation_has_gone_vague=False,
):
    """
    Full flow, in order:
    1. If a clarification was pending from the LAST turn, combine this
       reply with the original vague question and answer for real
       (via chat_history) - never touches the DAG.
    2. If the conversation has EVER gone vague before, skip straight to
       the chat_history-based answer path (DAG permanently bypassed for
       this conversation).
    3. Otherwise, classify this question as VAGUE or SPECIFIC.
       - VAGUE: flip conversation_has_gone_vague, ask a clarifying
         question (no DAG).
       - SPECIFIC: check the DAG (scoped to dag_current_node's children).
         Hit -> cached answer, move dag_current_node. Miss -> answer via
         retrieval+generation WITHOUT chat_history, store as a new DAG
         node, move dag_current_node.

    Returns a dict with: answer, chat_history, turn_number,
    retrieved_chunks_pool, pending_clarification, dag_current_node,
    conversation_has_gone_vague, already_streamed (pass all except
    already_streamed back in on the next call).
    """
    if chat_history is None:
        chat_history = []
    if retrieved_chunks_pool is None:
        retrieved_chunks_pool = {}

    # ---- Case A: resolving a pending clarification - always chat_history path, never DAG ----
    # handle_student_doubt() streams this live via generate_answer().
    if pending_clarification is not None:
        combined_question = f"{pending_clarification} (Specifically: {student_input})"
        result = handle_student_doubt(combined_question, chat_history, turn_number, retrieved_chunks_pool)
        return {
            "answer": result["answer"],
            "chat_history": result["chat_history"],
            "turn_number": result["turn_number"],
            "retrieved_chunks_pool": result["retrieved_chunks_pool"],
            "pending_clarification": None,
            "dag_current_node": dag_current_node,  # unchanged - DAG untouched
            "conversation_has_gone_vague": conversation_has_gone_vague,
            "already_streamed": True,  # generate_answer() already printed it live
        }

    # ---- Case B: conversation has EVER gone vague - DAG permanently bypassed ----
    if conversation_has_gone_vague:
        cached_answer = check_cache(student_input)  # flat semantic cache still allowed here
        if cached_answer is not None:
            print("[app] >>> ANSWERED FROM FLAT CACHE (no LLM call made) <<<")
            chat_history = chat_history + [
                {"role": "user", "content": student_input},
                {"role": "assistant", "content": cached_answer},
            ]
            return {
                "answer": cached_answer,
                "chat_history": chat_history,
                "turn_number": turn_number,
                "retrieved_chunks_pool": retrieved_chunks_pool,
                "pending_clarification": None,
                "dag_current_node": dag_current_node,
                "conversation_has_gone_vague": conversation_has_gone_vague,
                "already_streamed": False,  # cached string, needs simulated streaming
            }

        print("[app] >>> ANSWERED VIA chat_history (conversation flagged vague earlier) <<<")
        result = handle_student_doubt(student_input, chat_history, turn_number, retrieved_chunks_pool)
        return {
            "answer": result["answer"],
            "chat_history": result["chat_history"],
            "turn_number": result["turn_number"],
            "retrieved_chunks_pool": result["retrieved_chunks_pool"],
            "pending_clarification": None,
            "dag_current_node": dag_current_node,  # unchanged
            "conversation_has_gone_vague": conversation_has_gone_vague,
            "already_streamed": True,  # generate_answer() already printed it live
        }

    # ---- Case C: fresh question, conversation still "clean" - classify it ----
    state: ConversationState = {
        "student_input": student_input,
        "chat_history": chat_history,
        "is_vague": False,
        "clarifying_question": None,
        "awaiting_clarification": False,
        "final_answer": None,
    }
    state = detect_vague_question(state)

    if state["is_vague"]:
        # Flip the flag - DAG bypassed for the REST of this conversation
        conversation_has_gone_vague = True

        state = generate_clarifying_question(state)
        chat_history = chat_history + [
            {"role": "user", "content": student_input},
            {"role": "assistant", "content": state["clarifying_question"]},
        ]
        return {
            "answer": state["clarifying_question"],
            "chat_history": chat_history,
            "turn_number": turn_number,
            "retrieved_chunks_pool": retrieved_chunks_pool,
            "pending_clarification": student_input,
            "dag_current_node": dag_current_node,
            "conversation_has_gone_vague": conversation_has_gone_vague,
            "already_streamed": False,  # clarifying question text, needs simulated streaming
        }

    # ---- Case D: SPECIFIC question, conversation still clean - use the DAG ----
    match = find_matching_child(dag_current_node, student_input)

    if match is not None:
        print("[app] >>> ANSWERED FROM DAG CACHE (no LLM call made) <<<")
        chat_history = chat_history + [
            {"role": "user", "content": student_input},
            {"role": "assistant", "content": match["answer"]},
        ]
        return {
            "answer": match["answer"],
            "chat_history": chat_history,
            "turn_number": turn_number,
            "retrieved_chunks_pool": retrieved_chunks_pool,
            "pending_clarification": None,
            "dag_current_node": match["id"],  # move position to the matched node
            "conversation_has_gone_vague": conversation_has_gone_vague,
            "already_streamed": False,  # DAG-cached string, needs simulated streaming
        }

    # DAG miss - answer via retrieval + generation WITHOUT chat_history,
    # so the resulting cached node stays valid for any future student.
    # handle_student_doubt() streams this live via generate_answer().
    print("[app] >>> DAG MISS - answering via LIVE LLM PIPELINE (no chat_history), creating new DAG node <<<")
    result = handle_student_doubt(student_input, chat_history=[], turn_number=0, retrieved_chunks_pool={})
    new_node_id = create_node(student_input, result["answer"], parent_id=dag_current_node)

    # chat_history for the STUDENT's session still gets updated normally,
    # even though the DAG node itself was generated context-free.
    chat_history = chat_history + [
        {"role": "user", "content": student_input},
        {"role": "assistant", "content": result["answer"]},
    ]

    return {
        "answer": result["answer"],
        "chat_history": chat_history,
        "turn_number": turn_number + 1,
        "retrieved_chunks_pool": retrieved_chunks_pool,
        "pending_clarification": None,
        "dag_current_node": new_node_id,
        "conversation_has_gone_vague": conversation_has_gone_vague,
        "already_streamed": True,  # generate_answer() already printed it live
    }


def run_interactive():
    """Type your own questions live. See app.py's docstring for the
    full routing rules (clarification / vague-flag / DAG / chat_history)
    and the DISPLAY section for streaming behavior."""
    chat_history = None
    turn_number = 0
    pool = None
    pending_clarification = None
    dag_current_node = ROOT_NODE_ID
    conversation_has_gone_vague = False

    print("Kinematics doubt-solver (DAG cache + clarifying questions) — type 'quit' to exit\n")
    while True:
        student_input = input("You: ").strip()
        if student_input.lower() in ("quit", "exit"):
            break

        result = handle_student_doubt_with_clarification(
            student_input,
            chat_history,
            turn_number,
            pool,
            pending_clarification,
            dag_current_node,
            conversation_has_gone_vague,
        )

        if result["already_streamed"]:
            # generate_answer() already printed this live during the call above -
            # just add a newline for spacing, don't print the answer again.
            print()
        else:
            # Cache hit / clarifying question - simulate streaming for consistency.
            print("\nBot: ", end="")
            print_streamed(result["answer"])
            print()

        chat_history = result["chat_history"]
        turn_number = result["turn_number"]
        pool = result["retrieved_chunks_pool"]
        pending_clarification = result["pending_clarification"]
        dag_current_node = result["dag_current_node"]
        conversation_has_gone_vague = result["conversation_has_gone_vague"]


# ============================================================
# ASYNC VERSION — for FastAPI backend (jeeai-backend/main.py).
# Mirrors handle_student_doubt_with_clarification's routing logic
# exactly, but async throughout. Redis-backed cache functions
# (check_cache, find_matching_child, create_node) are synchronous,
# so they're run via asyncio.to_thread to avoid blocking.
# ============================================================

async def handle_student_doubt_with_clarification_async(
    student_input,
    chat_history=None,
    turn_number=0,
    retrieved_chunks_pool=None,
    pending_clarification=None,
    dag_current_node=ROOT_NODE_ID,
    conversation_has_gone_vague=False,
):
    if chat_history is None:
        chat_history = []
    if retrieved_chunks_pool is None:
        retrieved_chunks_pool = {}

    # ---- Case A: resolving a pending clarification ----
    if pending_clarification is not None:
        combined_question = f"{pending_clarification} (Specifically: {student_input})"
        result = await handle_student_doubt_async(combined_question, chat_history, turn_number, retrieved_chunks_pool)
        return {
            "answer": result["answer"],
            "chat_history": result["chat_history"],
            "turn_number": result["turn_number"],
            "retrieved_chunks_pool": result["retrieved_chunks_pool"],
            "pending_clarification": None,
            "dag_current_node": dag_current_node,
            "conversation_has_gone_vague": conversation_has_gone_vague,
        }

    # ---- Case B: conversation has EVER gone vague — DAG permanently bypassed ----
    if conversation_has_gone_vague:
        cached_answer = await asyncio.to_thread(check_cache, student_input)
        if cached_answer is not None:
            print("[app-async] >>> ANSWERED FROM FLAT CACHE (no LLM call made) <<<")
            chat_history = chat_history + [
                {"role": "user", "content": student_input},
                {"role": "assistant", "content": cached_answer},
            ]
            return {
                "answer": cached_answer,
                "chat_history": chat_history,
                "turn_number": turn_number,
                "retrieved_chunks_pool": retrieved_chunks_pool,
                "pending_clarification": None,
                "dag_current_node": dag_current_node,
                "conversation_has_gone_vague": conversation_has_gone_vague,
            }

        print("[app-async] >>> ANSWERED VIA chat_history (conversation flagged vague earlier) <<<")
        result = await handle_student_doubt_async(student_input, chat_history, turn_number, retrieved_chunks_pool)
        return {
            "answer": result["answer"],
            "chat_history": result["chat_history"],
            "turn_number": result["turn_number"],
            "retrieved_chunks_pool": result["retrieved_chunks_pool"],
            "pending_clarification": None,
            "dag_current_node": dag_current_node,
            "conversation_has_gone_vague": conversation_has_gone_vague,
        }

    # ---- Case C: fresh question, conversation still "clean" — classify it ----
    state: ConversationState = {
        "student_input": student_input,
        "chat_history": chat_history,
        "is_vague": False,
        "clarifying_question": None,
        "awaiting_clarification": False,
        "final_answer": None,
    }
    state = await detect_vague_question_async(state)

    if state["is_vague"]:
        conversation_has_gone_vague = True
        state = await generate_clarifying_question_async(state)
        chat_history = chat_history + [
            {"role": "user", "content": student_input},
            {"role": "assistant", "content": state["clarifying_question"]},
        ]
        return {
            "answer": state["clarifying_question"],
            "chat_history": chat_history,
            "turn_number": turn_number,
            "retrieved_chunks_pool": retrieved_chunks_pool,
            "pending_clarification": student_input,
            "dag_current_node": dag_current_node,
            "conversation_has_gone_vague": conversation_has_gone_vague,
        }

    # ---- Case D: SPECIFIC question, conversation still clean — use the DAG ----
    match = await asyncio.to_thread(find_matching_child, dag_current_node, student_input)

    if match is not None:
        print("[app-async] >>> ANSWERED FROM DAG CACHE (no LLM call made) <<<")
        chat_history = chat_history + [
            {"role": "user", "content": student_input},
            {"role": "assistant", "content": match["answer"]},
        ]
        return {
            "answer": match["answer"],
            "chat_history": chat_history,
            "turn_number": turn_number,
            "retrieved_chunks_pool": retrieved_chunks_pool,
            "pending_clarification": None,
            "dag_current_node": match["id"],
            "conversation_has_gone_vague": conversation_has_gone_vague,
        }

    print("[app-async] >>> DAG MISS - answering via LIVE LLM PIPELINE (no chat_history), creating new DAG node <<<")
    result = await handle_student_doubt_async(student_input, chat_history=[], turn_number=0, retrieved_chunks_pool={})
    new_node_id = await asyncio.to_thread(create_node, student_input, result["answer"], dag_current_node)

    chat_history = chat_history + [
        {"role": "user", "content": student_input},
        {"role": "assistant", "content": result["answer"]},
    ]

    return {
        "answer": result["answer"],
        "chat_history": chat_history,
        "turn_number": turn_number + 1,
        "retrieved_chunks_pool": retrieved_chunks_pool,
        "pending_clarification": None,
        "dag_current_node": new_node_id,
        "conversation_has_gone_vague": conversation_has_gone_vague,
    }


# ============================================================
# STREAMING VERSION — for real SSE (step 2.7). Mirrors the async
# version's routing logic, but is an async generator: yields
# {"type": "delta", "text": "..."} chunks as the answer becomes
# available — REAL live deltas for fresh LLM generations (via
# handle_student_doubt_stream_async), and cosmetic word-by-word
# chunks (20ms pacing, same as print_streamed) for already-complete
# text (cache hits, clarifying questions). Yields exactly ONE final
# {"type": "final", "result": {...}} at the end.
#
# UPDATED (classification): Case C now uses classify_question_type_async
# (VAGUE / CONTEXTUAL / SPECIFIC), instead of the old binary detect_vague
# check, so follow-ups that reference earlier turns (e.g. "explain
# the example you mentioned") get answered using real chat_history
# instead of being misjudged as vague just because the classifier
# couldn't see the conversation.
#
# UPDATED (DAG): Case D now also checks find_matching_node_globally
# on a local miss, BEFORE generating a fresh answer - if this exact
# question already exists ANYWHERE ELSE in the tree (reached via a
# different conversation path), the current position gets linked as
# an ADDITIONAL parent of that existing node (true multi-parent DAG)
# instead of creating a duplicate node with its own fresh LLM call.
# ============================================================

async def _yield_words_async(text, delay=0.02):
    """Cosmetic word-by-word streaming for already-complete text
    (cache hits, clarifying questions) — NOT real generation."""
    words = text.split(" ")
    for i, word in enumerate(words):
        piece = word + (" " if i < len(words) - 1 else "")
        yield {"type": "delta", "text": piece}
        await asyncio.sleep(delay)


async def handle_student_doubt_with_clarification_stream_async(
    student_input,
    chat_history=None,
    turn_number=0,
    retrieved_chunks_pool=None,
    pending_clarification=None,
    dag_current_node=ROOT_NODE_ID,
    conversation_has_gone_vague=False,
):
    if chat_history is None:
        chat_history = []
    if retrieved_chunks_pool is None:
        retrieved_chunks_pool = {}

    # ---- Case A: resolving a pending clarification — REAL streaming ----
    if pending_clarification is not None:
        combined_question = f"{pending_clarification} (Specifically: {student_input})"
        final_result = None
        async for piece in handle_student_doubt_stream_async(combined_question, chat_history, turn_number, retrieved_chunks_pool):
            if piece["type"] == "delta":
                yield {"type": "delta", "text": piece["text"]}
            else:
                final_result = piece["result"]

        yield {
            "type": "final",
            "result": {
                "answer": final_result["answer"],
                "chat_history": final_result["chat_history"],
                "turn_number": final_result["turn_number"],
                "retrieved_chunks_pool": final_result["retrieved_chunks_pool"],
                "pending_clarification": None,
                "dag_current_node": dag_current_node,  # unchanged - DAG untouched
                "conversation_has_gone_vague": conversation_has_gone_vague,
            },
        }
        return

    # ---- Case B: conversation has EVER gone vague — DAG permanently bypassed ----
    if conversation_has_gone_vague:
        cached_answer = await asyncio.to_thread(check_cache, student_input)
        if cached_answer is not None:
            print("[app-stream] >>> ANSWERED FROM FLAT CACHE (no LLM call made) <<<")
            async for piece in _yield_words_async(cached_answer):
                yield piece

            chat_history = chat_history + [
                {"role": "user", "content": student_input},
                {"role": "assistant", "content": cached_answer},
            ]
            yield {
                "type": "final",
                "result": {
                    "answer": cached_answer,
                    "chat_history": chat_history,
                    "turn_number": turn_number,
                    "retrieved_chunks_pool": retrieved_chunks_pool,
                    "pending_clarification": None,
                    "dag_current_node": dag_current_node,
                    "conversation_has_gone_vague": conversation_has_gone_vague,
                },
            }
            return

        print("[app-stream] >>> ANSWERED VIA chat_history (conversation flagged vague earlier) <<<")
        final_result = None
        async for piece in handle_student_doubt_stream_async(student_input, chat_history, turn_number, retrieved_chunks_pool):
            if piece["type"] == "delta":
                yield {"type": "delta", "text": piece["text"]}
            else:
                final_result = piece["result"]

        yield {
            "type": "final",
            "result": {
                "answer": final_result["answer"],
                "chat_history": final_result["chat_history"],
                "turn_number": final_result["turn_number"],
                "retrieved_chunks_pool": final_result["retrieved_chunks_pool"],
                "pending_clarification": None,
                "dag_current_node": dag_current_node,  # unchanged
                "conversation_has_gone_vague": conversation_has_gone_vague,
            },
        }
        return

    # ---- Case C: fresh question, conversation still "clean" — classify it
    # into VAGUE / CONTEXTUAL / SPECIFIC (context-aware, unlike the old
    # detect_vague_question_async which only ever saw the isolated message) ----
    question_type = await classify_question_type_async(student_input, chat_history)

    if question_type == "VAGUE":
        conversation_has_gone_vague = True
        state: ConversationState = {
            "student_input": student_input,
            "chat_history": chat_history,
            "is_vague": True,
            "clarifying_question": None,
            "awaiting_clarification": False,
            "final_answer": None,
        }
        state = await generate_clarifying_question_async(state)

        async for piece in _yield_words_async(state["clarifying_question"]):
            yield piece

        chat_history = chat_history + [
            {"role": "user", "content": student_input},
            {"role": "assistant", "content": state["clarifying_question"]},
        ]
        yield {
            "type": "final",
            "result": {
                "answer": state["clarifying_question"],
                "chat_history": chat_history,
                "turn_number": turn_number,
                "retrieved_chunks_pool": retrieved_chunks_pool,
                "pending_clarification": student_input,
                "dag_current_node": dag_current_node,
                "conversation_has_gone_vague": conversation_has_gone_vague,
            },
        }
        return

    if question_type == "CONTEXTUAL":
        # Refers back to something earlier in THIS conversation - answer
        # using real chat_history so the reference resolves correctly.
        # Do NOT permanently flip conversation_has_gone_vague, and do NOT
        # touch dag_current_node - so a LATER standalone question can
        # still hit the DAG cache normally.
        print("[app-stream] >>> CONTEXTUAL follow-up - answering via chat_history (DAG untouched) <<<")
        final_result = None
        async for piece in handle_student_doubt_stream_async(student_input, chat_history, turn_number, retrieved_chunks_pool):
            if piece["type"] == "delta":
                yield {"type": "delta", "text": piece["text"]}
            else:
                final_result = piece["result"]

        yield {
            "type": "final",
            "result": {
                "answer": final_result["answer"],
                "chat_history": final_result["chat_history"],
                "turn_number": final_result["turn_number"],
                "retrieved_chunks_pool": final_result["retrieved_chunks_pool"],
                "pending_clarification": None,
                "dag_current_node": dag_current_node,  # unchanged
                "conversation_has_gone_vague": conversation_has_gone_vague,  # unchanged (still False)
            },
        }
        return

    # ---- Case D: SPECIFIC question, conversation still clean — use the DAG ----
    match = await asyncio.to_thread(find_matching_child, dag_current_node, student_input)

    if match is not None:
        print("[app-stream] >>> ANSWERED FROM DAG CACHE (no LLM call made) <<<")
        async for piece in _yield_words_async(match["answer"]):
            yield piece

        chat_history = chat_history + [
            {"role": "user", "content": student_input},
            {"role": "assistant", "content": match["answer"]},
        ]
        yield {
            "type": "final",
            "result": {
                "answer": match["answer"],
                "chat_history": chat_history,
                "turn_number": turn_number,
                "retrieved_chunks_pool": retrieved_chunks_pool,
                "pending_clarification": None,
                "dag_current_node": match["id"],
                "conversation_has_gone_vague": conversation_has_gone_vague,
            },
        }
        return

    # NEW: no local match, but check if this exact question already
    # exists ANYWHERE ELSE in the tree before generating a fresh
    # answer - reuse it (true multi-parent DAG) instead of duplicating.
    global_match = await asyncio.to_thread(find_matching_node_globally, student_input)

    if global_match is not None:
        print("[app-stream] >>> ANSWERED FROM GLOBAL DAG MATCH (no LLM call, linking existing node) <<<")
        await asyncio.to_thread(add_child_edge, dag_current_node, global_match["id"])

        async for piece in _yield_words_async(global_match["answer"]):
            yield piece

        chat_history = chat_history + [
            {"role": "user", "content": student_input},
            {"role": "assistant", "content": global_match["answer"]},
        ]
        yield {
            "type": "final",
            "result": {
                "answer": global_match["answer"],
                "chat_history": chat_history,
                "turn_number": turn_number,
                "retrieved_chunks_pool": retrieved_chunks_pool,
                "pending_clarification": None,
                "dag_current_node": global_match["id"],
                "conversation_has_gone_vague": conversation_has_gone_vague,
            },
        }
        return

    print("[app-stream] >>> DAG MISS - answering via LIVE STREAMING LLM PIPELINE (no chat_history), creating new DAG node <<<")
    final_result = None
    async for piece in handle_student_doubt_stream_async(student_input, chat_history=[], turn_number=0, retrieved_chunks_pool={}):
        if piece["type"] == "delta":
            yield {"type": "delta", "text": piece["text"]}
        else:
            final_result = piece["result"]

    full_answer = final_result["answer"]
    new_node_id = await asyncio.to_thread(create_node, student_input, full_answer, dag_current_node)

    chat_history = chat_history + [
        {"role": "user", "content": student_input},
        {"role": "assistant", "content": full_answer},
    ]

    yield {
        "type": "final",
        "result": {
            "answer": full_answer,
            "chat_history": chat_history,
            "turn_number": turn_number + 1,
            "retrieved_chunks_pool": retrieved_chunks_pool,
            "pending_clarification": None,
            "dag_current_node": new_node_id,
            "conversation_has_gone_vague": conversation_has_gone_vague,
        },
    }


if __name__ == "__main__":
    run_interactive()