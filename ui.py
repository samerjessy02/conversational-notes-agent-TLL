"""
Minimal chat UI for the note-taking agent.

Destructive actions (modify/delete) pause the graph via LangGraph's
interrupt()/Command(resume=...) mechanism -- this is real human-in-the-loop,
not text parsing. When app.invoke() returns a "__interrupt__" key, we stop
showing the normal chat box and render actual Yes/No buttons instead. The
graph only proceeds once one of those buttons calls
app.invoke(Command(resume={"approved": bool}), config=...).

Under each reply there's also a "Tool calls & intent" expander showing the
classified intent (a hint only now, not load-bearing) and every tool call
made that turn, for debugging.

Run with:  streamlit run ui.py
"""
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agent import build_app

st.set_page_config(page_title="Note Agent", page_icon="🗒️", layout="centered")


@st.cache_resource
def get_app():
    return build_app()


try:
    app, db = get_app()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

CONFIG = {"configurable": {"thread_id": "ui-session"}}

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None
if "turn_prior_count" not in st.session_state:
    st.session_state.turn_prior_count = 0

st.title("🗒️ Note-Taking Agent")
st.caption('Try: "add a note about groceries", "what did I write about the API?", "delete the office note"')
st.sidebar.caption(f"Notes DB: `{db.db_path}` (persists across restarts)")

show_debug = st.sidebar.checkbox("Show tool-call trace under each reply", value=True)
if st.sidebar.button("Reset conversation"):
    st.session_state.history = []
    st.session_state.pending_interrupt = None
    get_app.clear()
    st.rerun()

with st.sidebar.expander("📋 Current notes (debug)", expanded=False):
    notes = db.search_notes()
    if notes:
        for n in notes:
            st.markdown(f"**#{n['id']} {n['title']}** — _{', '.join(n['tags']) or 'no tags'}_")
    else:
        st.caption("No notes.")


def render_trace(trace: dict) -> None:
    st.markdown(f"**Detected intent:** `{trace['intent']}`")
    if not trace["steps"]:
        st.caption("No tool calls this turn.")
        return
    for step in trace["steps"]:
        if step["kind"] == "tool_call":
            st.code(f"{step['name']}({step['args']})", language="python")
        else:
            st.text(step["content"])


def extract_trace(result: dict, prior_count: int):
    new_messages = result["messages"][prior_count:]
    steps = []
    final_text = ""
    for m in new_messages:
        if isinstance(m, AIMessage):
            if m.tool_calls:
                for tcall in m.tool_calls:
                    steps.append({"kind": "tool_call", "name": tcall["name"], "args": tcall["args"]})
            if m.content:
                final_text = m.content
        elif isinstance(m, ToolMessage):
            steps.append({"kind": "tool_result", "content": m.content})
    return final_text, steps


def handle_result(result: dict, prior_count: int) -> None:
    if "__interrupt__" in result:
        st.session_state.pending_interrupt = result["__interrupt__"][0].value
        st.session_state.turn_prior_count = prior_count
        return
    st.session_state.pending_interrupt = None
    final_text, steps = extract_trace(result, prior_count)
    trace = {"intent": result.get("last_intent"), "steps": steps}
    st.session_state.history.append(
        {"role": "assistant", "content": final_text or "(no reply)", "trace": trace}
    )


# Replay prior turns
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn["role"] == "assistant" and show_debug and turn.get("trace"):
            with st.expander("🔧 Tool calls & intent"):
                render_trace(turn["trace"])

# A destructive action is awaiting approval -- show buttons, not free text
if st.session_state.pending_interrupt:
    payload = st.session_state.pending_interrupt
    changes = payload.get("changes", [])
    with st.chat_message("assistant"):
        if len(changes) == 1:
            st.warning(changes[0]["summary"])
        else:
            st.warning("Multiple actions need your approval:")
            for c in changes:
                st.markdown(f"- {c['summary']}")
        col1, col2 = st.columns(2)
        yes_clicked = col1.button("✅ Yes, proceed", use_container_width=True, key="confirm_yes")
        no_clicked = col2.button("❌ No, cancel", use_container_width=True, key="confirm_no")
        if yes_clicked or no_clicked:
            with st.spinner("Applying your decision..."):
                result = app.invoke(Command(resume={"approved": yes_clicked}), config=CONFIG)
            handle_result(result, st.session_state.turn_prior_count)
            st.rerun()

user_input = st.chat_input(
    "Ask me to add, find, update, or delete a note..." if not st.session_state.pending_interrupt
    else "Respond to the confirmation above first",
    disabled=bool(st.session_state.pending_interrupt),
)
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    prior_snapshot = app.get_state(CONFIG)
    prior_count = len(prior_snapshot.values.get("messages", [])) if prior_snapshot.values else 0

    with st.spinner("Thinking..."):
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)

    handle_result(result, prior_count)
    st.rerun()
