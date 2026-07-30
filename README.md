# Conversational Note-Taking Agent

A small LangGraph agent that manages notes through chat: add, search, update,
delete, all through natural language, backed by Groq for the LLM.

## Project layout

```
agent.py               core logic: DB, tools, intent classifier, graph
main.py                CLI entrypoint
ui.py                  Streamlit chat UI with a debug/trace panel
test_agent_smoke.py    smoke test using a stubbed LLM (no API key needed)
requirements.txt
```

`agent.py` is the only file that actually matters architecturally — both
`main.py` and `ui.py` just call `build_app()` from it and drive the graph.

## How it's put together

**The database** is an in-memory dict (`NoteDatabase`) — no persistence, it
resets every time you restart. Fine for a demo, would need to be swapped for
something like SQLite if this ever needed to survive a restart.

**Tools** wrap the database for the LLM: `add_note`, `search_notes`,
`modify_note`, `delete_note`. The two destructive ones don't actually touch
the database. They look up the note, describe what would change, and stash
that proposal into the graph's state. Nothing gets deleted or edited at that
point — it's just a pending action sitting in state.

**The confirmation step is not something the LLM controls.** Earlier versions
of this had a `confirmed: bool` argument the model itself was responsible
for setting correctly, which is a fragile thing to rely on — a model can
just set it to `True` on the first call, or mix up which note it's
confirming. Instead there's a separate graph node,
`execute_confirmation_node`, and it's the only code path that ever writes to
the database for a modify/delete. The graph only routes there when a plain
"yes" or "no" is detected in the user's next message — that detection is a
regex, not a model call, so it can't be talked out of it.

**The intent classifier** (`classify_intent`) is also just regex/keyword
matching — it tags each message as create / search / modify / delete /
confirm_yes / confirm_no / chitchat. For the first four it's just a hint
added to the system prompt, the model can ignore it. For confirm_yes /
confirm_no it's load-bearing: that's what triggers the deterministic
execution path above. It's deliberately simple rather than a second LLM
call — instant, free, and you can see exactly why it classified something
a certain way in the UI's trace panel.

**Graph flow**, roughly:

```
classify_intent → (confirm_yes/no + something pending?) → execute_confirmation → end
                → (anything else) → agent → tool call? → tools → back to agent → end
```

## Running it

Using `uv`:

```bash
uv venv
uv pip install -r requirements.txt
```

Add your Groq key — either export it or drop a `.env` file in the project
root:

```
GROQ_API_KEY=your_key_here
```

Then run whichever entrypoint you want:

```bash
uv run python main.py              # terminal chat
uv run streamlit run ui.py         # browser chat with a tool-call trace panel
uv run python test_agent_smoke.py  # sanity check, no API key required
```

If you're not using `uv`, the equivalent is a normal venv + `pip install -r
requirements.txt`, then `python main.py` / `streamlit run ui.py`.

## The UI

`streamlit run ui.py` gives you a plain chat box. Under each reply there's a
"Tool calls & intent" expander showing what got classified and which tool
was actually called with which arguments — useful when something doesn't do
what you expected and you want to know whether it's a search problem, a
routing problem, or the model just picked the wrong tool. The sidebar also
has a live dump of the notes currently in memory and a reset button.

It's single-session by design (one fixed thread id, one shared in-memory
DB) — good for testing locally, not built for multiple concurrent users.

## Known gaps

- No persistence — restarting wipes the notes.
- The intent classifier is a heuristic and will get phrasing it hasn't seen
  wrong sometimes. That's acceptable everywhere except the yes/no
  confirmation check, which is why that specific case has a smoke test.
- If two destructive actions get proposed back to back before either is
  confirmed, the second proposal just overwrites the first — there's no
  queue.