import asyncio
import app

async def main():
    print("Orchestration streaming test:\n")
    async for piece in app.handle_student_doubt_with_clarification_stream_async("What is average velocity?"):
        if piece["type"] == "delta":
            print(piece["text"], end="", flush=True)
        elif piece["type"] == "final":
            print("\n\n--- FINAL RESULT ---")
            print("DAG node:", piece["result"]["dag_current_node"])
            print("Conversation vague:", piece["result"]["conversation_has_gone_vague"])

asyncio.run(main())