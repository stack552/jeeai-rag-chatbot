"""
langgraph_clarify.py
----------------------
LangGraph-based clarifying-question flow: if a student asks something
too vague (e.g. "I don't understand this lecture"), the chatbot asks
a clarifying question (which lecture / which concept) instead of
guessing.

SELF-CONTAINED: does NOT import from rag_pipeline.py, to avoid a
circular import (rag_pipeline / app.py needs to import FROM this file).
Has its own langchain_llm instance instead.

Provides: detect_vague_question(), generate_clarifying_question().
The orchestration that combines these with real retrieval/generation
(handle_student_doubt from rag_pipeline.py) lives in app.py.

BEFORE RUNNING:
Set your Groq key as an environment variable (never hardcoded here -
this file is safe to keep in a public GitHub repo as-is):
    set GROQ_API_KEY=your_actual_key_here      (Windows CMD, per session)
"""

import os
from typing import TypedDict, Optional
from langchain_groq import ChatGroq

# ---------- CONFIG ----------
# Reads your Groq key from an environment variable - never hardcoded here,
# so this file is safe to commit and push to a public GitHub repo.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable not set. Run this first: "
        "set GROQ_API_KEY=your_actual_key_here"
    )
# -----------------------------

langchain_llm = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY, temperature=0.3)


# ---------- State definition ----------
class ConversationState(TypedDict):
    student_input: str
    chat_history: list
    is_vague: bool
    clarifying_question: Optional[str]
    awaiting_clarification: bool
    final_answer: Optional[str]


# ---------- Vague-question detector node ----------
def detect_vague_question(state: ConversationState) -> ConversationState:
    """
    Classifies whether the student's question is too vague to answer
    directly (needs clarification) or specific enough to retrieve and
    answer right away.
    """
    prompt = f"""Is this student's physics question too vague to answer
directly (e.g. "I don't understand this lecture", "explain this",
"I'm confused"), or is it specific enough to answer directly (mentions
a formula, a specific concept, or a specific part of a problem)?

Question: {state['student_input']}

Reply with ONLY one word: VAGUE or SPECIFIC."""

    result = langchain_llm.invoke(prompt).content.strip().upper()
    state["is_vague"] = (result == "VAGUE")
    return state


# ---------- Clarifying-question generator node ----------
def generate_clarifying_question(state: ConversationState) -> ConversationState:
    """
    Only runs when the question was classified as VAGUE. Asks the
    student a clarifying question instead of guessing what to answer.
    If recent chat_history exists, tries to NAME the specific options
    the student might mean (e.g. actual formulas or examples the bot
    just gave), rather than a generic "which topic" fallback.
    """
    if state["chat_history"]:
        recent = state["chat_history"][-2:]
        history_lines = []
        for msg in recent:
            role = "Student" if msg["role"] == "user" else "Bot"
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines)

        prompt = f"""The student said something vague or ambiguous: "{state['student_input']}"

Recent conversation:
{history_text}

The bot's last message above likely contains specific formulas, named
examples, or distinct cases. Your job is to ask which ONE of those
SPECIFIC items the student means.

STRICT RULES:
- You MUST quote or name at least one actual formula, number, or example from the conversation above, word for word.
- You are FORBIDDEN from asking "which lecture" or "which topic" - the student already told you the topic.
- Keep it to 2-3 sentences.

Example of a GOOD clarifying question (for illustration only, not this case):
"Do you mean the formula t = sqrt(2h/g) for a projectile launched from a height, or T = (2 v0 sin(theta))/g for a projectile that lands at the same height it was launched from?"

Now write a similarly specific clarifying question for THIS case."""
    else:
        prompt = f"""The student said something vague: "{state['student_input']}"

There is no prior conversation context. Ask the student which lecture
or topic in kinematics they need help with, so you can help them
properly. Keep it to 1-2 sentences."""

    clarifying_q = langchain_llm.invoke(prompt).content.strip()
    state["clarifying_question"] = clarifying_q
    state["final_answer"] = clarifying_q
    state["awaiting_clarification"] = True
    return state


# ============================================================
# ASYNC VERSIONS — for FastAPI backend, via app.py's async orchestrator.
# These sit alongside the sync versions above; run_interactive() in
# app.py keeps using the sync versions untouched.
# ============================================================

async def detect_vague_question_async(state: ConversationState) -> ConversationState:
    """Async version of detect_vague_question."""
    prompt = f"""Is this student's physics question too vague to answer
directly (e.g. "I don't understand this lecture", "explain this",
"I'm confused"), or is it specific enough to answer directly (mentions
a formula, a specific concept, or a specific part of a problem)?

Question: {state['student_input']}

Reply with ONLY one word: VAGUE or SPECIFIC."""

    response = await langchain_llm.ainvoke(prompt)
    result = response.content.strip().upper()
    state["is_vague"] = (result == "VAGUE")
    return state


async def generate_clarifying_question_async(state: ConversationState) -> ConversationState:
    """
    Async version of generate_clarifying_question. If recent
    chat_history exists, tries to NAME the specific options the
    student might mean (e.g. actual formulas or examples the bot
    just gave), rather than a generic "which topic" fallback.
    """
    if state["chat_history"]:
        recent = state["chat_history"][-2:]
        history_lines = []
        for msg in recent:
            role = "Student" if msg["role"] == "user" else "Bot"
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines)

        prompt = f"""The student said something vague or ambiguous: "{state['student_input']}"

Recent conversation:
{history_text}

The bot's last message above likely contains specific formulas, named
examples, or distinct cases. Your job is to ask which ONE of those
SPECIFIC items the student means.

STRICT RULES:
- You MUST quote or name at least one actual formula, number, or example from the conversation above, word for word.
- You are FORBIDDEN from asking "which lecture" or "which topic" - the student already told you the topic.
- Keep it to 2-3 sentences.

Example of a GOOD clarifying question (for illustration only, not this case):
"Do you mean the formula t = sqrt(2h/g) for a projectile launched from a height, or T = (2 v0 sin(theta))/g for a projectile that lands at the same height it was launched from?"

Now write a similarly specific clarifying question for THIS case."""
    else:
        prompt = f"""The student said something vague: "{state['student_input']}"

There is no prior conversation context. Ask the student which lecture
or topic in kinematics they need help with, so you can help them
properly. Keep it to 1-2 sentences."""

    response = await langchain_llm.ainvoke(prompt)
    clarifying_q = response.content.strip()
    state["clarifying_question"] = clarifying_q
    state["final_answer"] = clarifying_q
    state["awaiting_clarification"] = True
    return state


async def classify_question_type_async(student_input: str, chat_history: list) -> str:
    """
    Classifies the student's latest message, now taking the recent
    conversation into account (unlike detect_vague_question_async,
    which only ever saw the isolated message with no context).

    Returns one of:
      - "VAGUE" — too unclear to answer, even given the conversation so
        far, and nothing in it resolves what's being asked about.
      - "CONTEXTUAL" — refers back to something already discussed
        (e.g. "explain that in detail", "the example you mentioned"),
        AND the recent conversation actually contains enough
        information to resolve that reference.
      - "SPECIFIC" — self-contained; answerable on its own, without
        needing the conversation history.
    """
    if chat_history:
        recent = chat_history[-4:]
        history_lines = []
        for msg in recent:
            role = "Student" if msg["role"] == "user" else "Bot"
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines)
    else:
        history_text = "(no prior conversation)"

    prompt = f"""You are classifying a student's physics question to decide how to answer it.

Recent conversation:
{history_text}

Student's latest message: "{student_input}"

Classify into exactly ONE of these three categories:

VAGUE - too unclear or open-ended to answer, even considering the conversation so far, and the recent conversation does NOT resolve what is being asked about.

CONTEXTUAL - refers back to something already discussed in the recent conversation above (e.g. "explain that in detail", "the example you mentioned", "what about the second one"), AND the recent conversation DOES contain enough information to understand what is being referred to.

SPECIFIC - self-contained; can be understood and answered on its own, without needing the conversation history (it names a concept, formula, or specific scenario directly).

Reply with ONLY one word: VAGUE, CONTEXTUAL, or SPECIFIC."""

    response = await langchain_llm.ainvoke(prompt)
    result = response.content.strip().upper()
    if result not in ("VAGUE", "CONTEXTUAL", "SPECIFIC"):
        result = "SPECIFIC"  # safe fallback if the model replies with something unexpected
    return result


# ---------- Standalone test (classifier + clarifier only, no real answers) ----------
if __name__ == "__main__":
    test_cases = [
        {
            "student_input": "I can't understand this, can you explain please?",
            "chat_history": [],
        },
        {
            "student_input": "I don't get it, can you explain?",
            "chat_history": [
                {"role": "user", "content": "Why tau_A = 2*l*eta / (v0*(eta^2 - 1))?"},
                {"role": "assistant", "content": "This comes from the boat-river problem in Lecture 29..."},
            ],
        },
        {
            "student_input": "Why tau_A = 2*l*eta / (v0*(eta^2 - 1))?",
            "chat_history": [],
        },
    ]

    for case in test_cases:
        state: ConversationState = {
            "student_input": case["student_input"],
            "chat_history": case["chat_history"],
            "is_vague": False,
            "clarifying_question": None,
            "awaiting_clarification": False,
            "final_answer": None,
        }

        state = detect_vague_question(state)
        print(f"Question: {case['student_input']!r}")
        print(f"  is_vague: {state['is_vague']}")

        if state["is_vague"]:
            state = generate_clarifying_question(state)
            print(f"  Clarifying question asked: {state['clarifying_question']}")
        else:
            print("  -> Would proceed to normal retrieve_and_answer (see app.py)")
        print()