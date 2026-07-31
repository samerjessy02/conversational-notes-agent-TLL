from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agent import build_app


def main():
    app, db = build_app()
    config = {"configurable": {"thread_id": "session_1"}}

    print("=" * 60)
    print(" Conversational Note-Taking Agent (Groq Powered)")
    print(" Try asking:")
    print("   - 'What did I write about the API?'")
    print("   - 'Update my standup note to say Wednesdays'")
    print("   - 'Delete the note about the old office address'")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input or user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

            # A destructive tool call pauses the graph via interrupt(). Prompt
            # for y/n right here and resume -- nothing runs until we do.
            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                print("\n[CONFIRM]")
                for change in payload["changes"]:
                    print(f"  - {change['summary']}")
                answer = input("Proceed? (yes/no): ").strip().lower()
                approved = answer in ("y", "yes")
                result = app.invoke(Command(resume={"approved": approved}), config=config)

            last_msg = result["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                print(f"\nAgent: {last_msg.content}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
