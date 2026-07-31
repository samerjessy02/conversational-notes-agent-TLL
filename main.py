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

            prior_snapshot = app.get_state(config)
            seen_count = len(prior_snapshot.values.get("messages", [])) if prior_snapshot.values else 0

            def print_new_assistant_texts(result, seen_count):
                # Print every bit of assistant commentary as it appears,
                # including on a leg that immediately triggers ANOTHER
                # interrupt -- e.g. "Cancelled note #4, I'll try note #7 now"
                # would otherwise be silently lost.
                for m in result["messages"][seen_count:]:
                    if isinstance(m, AIMessage) and m.content:
                        print(f"\nAgent: {m.content}")
                return len(result["messages"])

            result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            seen_count = print_new_assistant_texts(result, seen_count)

            # A destructive tool call pauses the graph via interrupt(). Prompt
            # for y/n right here and resume -- nothing runs until we do. If
            # the model proposes another destructive action right after this
            # one is resolved, a FRESH interrupt fires and we loop again --
            # each one requires its own separate yes/no.
            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                print("\n[CONFIRM]")
                for change in payload["changes"]:
                    print(f"  - {change['summary']}")
                answer = input("Proceed? (yes/no): ").strip().lower()
                approved = answer in ("y", "yes")
                result = app.invoke(Command(resume={"approved": approved}), config=config)
                seen_count = print_new_assistant_texts(result, seen_count)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
