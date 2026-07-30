from langchain_core.messages import AIMessage, HumanMessage

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

            input_state = {"messages": [HumanMessage(content=user_input)]}

            for event in app.stream(input_state, config=config, stream_mode="values"):
                last_msg = event["messages"][-1]

            if isinstance(last_msg, AIMessage) and last_msg.content:
                print(f"\nAgent: {last_msg.content}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
