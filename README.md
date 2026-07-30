# Conversational Note-Taking Agent

## Files

| File | Purpose |
|---|---|
| `agent.py` | Everything shared: the mock DB, tools, intent classifier, system prompt, and the LangGraph workflow. Both entrypoints import `build_app()` from here. |
| `main.py` | Original CLI, now a thin wrapper around `agent.py`. |
| `ui.py` | New minimal Streamlit chat UI, with a per-reply "🔧 Tool calls & intent" panel for troubleshooting. |
| `test_agent_smoke.py` | Scripted-LLM smoke test proving the confirmation flow and intent classifier actually work, without needing a real API key. |

## Running it

```bash
pip install -r requirements.txt

# .env or exported:
GROQ_API_KEY=your_key_here

python main.py           # CLI
streamlit run ui.py      # Web UI
```

Run the smoke test any time (no API key needed, it stubs the LLM):

```bash
python test_agent_smoke.py
```

## What changed from your version, and why

### 1. Intent classifier
`classify_intent()` in `agent.py` is a small, fast, regex/keyword classifier
that tags every user message as `create` / `search` / `modify` / `delete` /
`confirm_yes` / `confirm_no` / `chitchat`. It's deliberately rule-based
rather than a second LLM call — free, instant, and its output is directly
visible in the UI's trace panel, which makes misroutes easy to debug. It's
used two ways:

- As a **hint** appended to the system prompt for create/search/modify/delete
  (the LLM can still override it — it's guidance, not a hard constraint).
- As a **hard routing decision** for `confirm_yes` / `confirm_no` — see below.

### 2. Confirmation is now enforced in code, not trusted to the LLM
This was the main structural risk in the original version. `modify_note_tool`
and `delete_note_tool` took a `confirmed: bool` argument that the *model
itself* supplied — nothing stopped the model from sending `confirmed=True`
on the very first call, or from confirming the wrong note if a fast/small
tool-calling model got two turns crossed.

Now:
- `modify_note_tool` / `delete_note_tool` **never touch the database**. They
  only look up the note and stage the proposed change into graph state
  (`pending_confirmation`) via a `Command` update, then return a
  `REQUIRES_CONFIRMATION` message.
- The **only** place a mutation actually happens is `execute_confirmation_node`
  in the graph, and the graph only reaches that node when `classify_intent()`
  deterministically reads "yes" or "no" from the user's *own next message* —
  the LLM is never asked to decide whether something is confirmed.
- There's no `confirmed` field anywhere for the model to set.

This means a destructive action always requires: propose (by tool call) →
a literal yes/no from the human → then, and only then, code that isn't an
LLM call touches the DB.

### 3. Bug fix: truthy checks in the confirmation summary
The original used `if title:` / `if tags:` when building the "proposed
changes" description, so intentionally clearing a field to `""` or `[]`
wouldn't show up as a change. Fixed to `is not None`.

### 4. Clear startup error
Missing `GROQ_API_KEY` now raises a readable error (`RuntimeError` in the
CLI, `st.error` in the UI) instead of a confusing stack trace on the first
LLM call.

### 5. Shared `agent.py`
DB/tools/graph were pulled out of `main.py` into `agent.py` so the CLI and
the new UI don't maintain two copies of the same logic.

## Minimal UI

`streamlit run ui.py` gives you:
- A plain chat interface.
- A sidebar toggle to show/hide the debug trace, a "Reset conversation"
  button, and a live dump of the current notes in the mock DB.
- Under each assistant reply, an expandable **"🔧 Tool calls & intent"**
  panel showing the classified intent, every tool call made that turn (name
  + args), and the raw tool result string — this is the "what actually
  happened" view for troubleshooting search misses, wrong note IDs, etc.

It's intentionally single-file and has no auth, no persistence beyond the
in-memory DB, and one shared conversation thread (`thread_id="ui-session"`)
— fine for local debugging, not for multiple concurrent users.

## Test cases

**Happy path**
- "Add a note titled 'Groceries', body 'milk, eggs', tag shopping" → created, no confirmation needed.
- "What did I write about the API?" → finds the API Redesign note.
- "Show my meeting notes" → tag-based or keyword search, depending on phrasing.
- "Update my standup note to say Wednesdays" → since 3 notes share the title "Team Standup", expect the agent to search, find multiple matches, and ask which one before proposing anything.
- "Delete the note about the old office address" → search finds it → proposes deletion → reply "yes" → deleted. Check the sidebar note list updates.
- Reply "no" to a proposed deletion/modification → nothing changes, `pending_confirmation` clears (visible in the trace panel as no further tool calls).

**Edge cases worth checking**
- **Ambiguous target**: "delete my standup note" with 3 same-titled notes — the agent should list IDs and ask, not guess. If it doesn't, that's a system-prompt tuning issue, not a graph bug (the graph won't let a delete through without an explicit note_id anyway).
- **Bare "yes" with nothing pending**: send "yes" with no prior proposal — should fall through to the normal agent path and just get a conversational reply, not crash.
- **Non-existent note**: "delete note 999" — tool returns `ERROR: Note #999 does not exist.` before anything is staged; confirming "yes" afterward should say there's nothing pending (since no `pending_confirmation` was ever set).
- **Confirmation phrased conversationally**: "yeah, go ahead" / "nah, don't" — should still resolve to confirm_yes/confirm_no (covered in `test_agent_smoke.py`).
- **Long sentence containing a yes/no word**: "no, delete note 3 instead" — the classifier gates yes/no detection to short replies (≤6 words) specifically so this doesn't get misread as a bare rejection; it should fall through to `delete` intent.
- **Race between two pending confirmations**: propose a delete, then before confirming, ask to modify a *different* note — the second proposal overwrites `pending_confirmation`, so a later "yes" applies to the second one only. Worth deciding if you want stacked/queued confirmations instead; current behavior is last-proposal-wins.
- **Search with relative time**: "what did I add last week" — the temporal-word stripping regex should leave just meaningful keywords; if the query becomes empty (e.g., user says only "yesterday"), it degenerates to `tag`-only or "list everything," worth checking that isn't confusing to the model.
- **Empty title**: "add a note with no title" — `add_note_tool` returns an explicit `ERROR: Title cannot be empty.` rather than silently creating a blank-titled note.

## Known limitations (didn't fix, flagging for visibility)
- Notes are still in-memory only (reset on restart) — fine for a demo, not for production. If you want this to survive restarts, the DB layer is the only thing that would need to change (swap `NoteDatabase` for a SQLite-backed repository with the same method signatures).
- The UI uses one global `db` and one fixed `thread_id`, so it's single-user/single-session by design.
- The intent classifier is a heuristic, not a model — it will misfire on unusual phrasing. It's a *hint* everywhere except the yes/no confirmation gate, which is the one place correctness actually matters, and that's the part the smoke test covers.
