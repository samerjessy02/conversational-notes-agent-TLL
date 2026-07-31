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
if "confirmation_round" not in st.session_state:
    # Bumped every time a NEW pending_interrupt is set, and folded into the
    # confirm/cancel button keys below. Streamlit identifies widgets by key,
    # and reusing a static key across multiple confirmations in the same
    # session is exactly the kind of ambiguity that causes a click to be
    # misattributed to a stale widget instance.
    st.session_state.confirmation_round = 0
if "accumulated_steps" not in st.session_state:
    # Tool-call/tool-result steps carry across a chain of interrupts within
    # one logical turn (e.g. propose -> rejected -> propose something else
    # -> approved), so the final trace shown to the user covers the whole
    # chain, not just the last leg of it.
    st.session_state.accumulated_steps = []

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


def handle_result(result: dict, prior_count: int) -> None:
    """Process one invoke()/resume() result. A single logical "turn" from the
    user's point of view can involve MULTIPLE calls to this function if the
    model proposes several destructive actions back to back (each gets its
    own interrupt). We must not lose any assistant commentary that shows up
    on an intermediate leg -- e.g. "Cancelled note #4. I'll delete note #7
    now" attached to the very message that triggers the *next* interrupt --
    so every AIMessage.content we see gets surfaced immediately, and tool
    call/result steps accumulate across the whole chain for the final trace.
    """
    new_messages = result["messages"][prior_count:]
    assistant_texts = []
    for m in new_messages:
        if isinstance(m, AIMessage):
            if m.tool_calls:
                for tcall in m.tool_calls:
                    st.session_state.accumulated_steps.append(
                        {"kind": "tool_call", "name": tcall["name"], "args": tcall["args"]}
                    )
            if m.content:
                assistant_texts.append(m.content)
        elif isinstance(m, ToolMessage):
            st.session_state.accumulated_steps.append({"kind": "tool_result", "content": m.content})

    if "__interrupt__" in result:
        # Show any commentary the model gave before pausing again, then wait
        # on the new confirmation -- nothing here is a "final" reply yet.
        for text in assistant_texts:
            st.session_state.history.append({"role": "assistant", "content": text, "trace": None})
        st.session_state.pending_interrupt = result["__interrupt__"][0].value
        st.session_state.turn_prior_count = len(result["messages"])
        st.session_state.confirmation_round += 1
        return

    st.session_state.pending_interrupt = None
    final_text = assistant_texts[-1] if assistant_texts else "(no reply)"
    trace = {"intent": result.get("last_intent"), "steps": st.session_state.accumulated_steps}
    st.session_state.history.append({"role": "assistant", "content": final_text, "trace": trace})
    st.session_state.accumulated_steps = []


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
        round_id = st.session_state.confirmation_round
        col1, col2 = st.columns(2)
        yes_clicked = col1.button("✅ Yes, proceed", use_container_width=True, key=f"confirm_yes_{round_id}")
        no_clicked = col2.button("❌ No, cancel", use_container_width=True, key=f"confirm_no_{round_id}")
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
