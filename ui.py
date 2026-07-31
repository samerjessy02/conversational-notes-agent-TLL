"""
Minimal chat UI for the note-taking agent, with a "Tool calls & intent"
expander under each reply so you can see exactly what happened on that
turn: the classified intent, which tool(s) were called with which args,
and the raw tool result -- useful for debugging things like "why didn't it
find my note" or "why did it think this was a delete request".

Run with:  streamlit run ui.py
"""
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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

st.title("🗒️ Note-Taking Agent")
st.caption('Try: "add a note about groceries", "what did I write about the API?", "delete the office note"')
st.sidebar.caption(f"Notes DB: `{db.db_path}` (persists across restarts)")

show_debug = st.sidebar.checkbox("Show tool-call trace under each reply", value=True)
if st.sidebar.button("Reset conversation"):
    st.session_state.history = []
    get_app.clear()
    st.rerun()

with st.sidebar.expander("📋 Current notes (debug)", expanded=False):
    notes = db.search_notes()
    if notes:
        for n in notes:
            st.markdown(f"**#{n['id']} {n['title']}** — _{', '.join(n['tags']) or 'no tags'}_")
    else:
        st.caption("No notes.")

with st.sidebar.expander("⏳ Pending confirmations (debug)", expanded=False):
    snapshot = app.get_state(CONFIG)
    pending = (snapshot.values.get("pending_confirmations") or []) if snapshot.values else []
    if pending:
        for i, p in enumerate(reversed(pending)):
            marker = "next to resolve" if i == 0 else "queued"
            st.markdown(f"- `{p['tool']}` on #{p['note_id']} ({marker}): {p['summary']}")
    else:
        st.caption("Nothing pending.")


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


# Replay prior turns
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn["role"] == "assistant" and show_debug and turn.get("trace"):
            with st.expander("🔧 Tool calls & intent"):
                render_trace(turn["trace"])

user_input = st.chat_input("Ask me to add, find, update, or delete a note...")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    prior_snapshot = app.get_state(CONFIG)
    prior_count = len(prior_snapshot.values.get("messages", [])) if prior_snapshot.values else 0

    with st.spinner("Thinking..."):
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)

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

    trace = {"intent": result.get("last_intent"), "steps": steps}
    final_text = final_text or "(no reply)"

    with st.chat_message("assistant"):
        st.write(final_text)
        if show_debug:
            with st.expander("🔧 Tool calls & intent"):
                render_trace(trace)

    st.session_state.history.append({"role": "assistant", "content": final_text, "trace": trace})
