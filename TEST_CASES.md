# Test Cases

Two kinds of tests here. The automated ones (`test_agent_smoke.py`) check
the mechanics — the interrupt/resume pause, the DB, persistence — using a
scripted fake LLM, so they're exact and repeatable. The manual ones exercise
the real model's tool-calling and phrasing judgement, which varies a bit
run to run — treat "expected" there as "what a reasonable model should do,"
not a byte-exact string match.

## Automated (`python test_agent_smoke.py`)

| Test | What it proves |
|---|---|
| `test_intent_classifier` | Classifier labels create/search/modify/delete/chitchat correctly (advisory only now, doesn't gate anything) |
| `test_persistence_across_reopen` | Notes added in one `NoteDatabase` instance are visible after closing and reopening a new instance on the same file path; reopening a non-empty file does not re-seed |
| `test_delete_persists_across_reopen` | A deletion made before a simulated restart is still gone after |
| `test_delete_requires_explicit_approval` | `delete_note` never touches the DB before `Command(resume={"approved": True})`; rejecting leaves the note untouched; approving actually deletes it |
| `test_modify_requires_explicit_approval` | Same guarantee for `modify_note` |
| `test_bundled_multi_change_turn_all_or_nothing` | Two destructive tool calls proposed in the same turn produce ONE confirmation payload with both changes listed; a single approval applies both |
| `test_rejecting_bundled_changes_applies_neither` | A single rejection cancels both bundled changes, not just one |

Run it any time with `python test_agent_smoke.py` (or `uv run python
test_agent_smoke.py`) — no API key required, it stubs the LLM.

## Manual (run against the real CLI or UI, with a real `GROQ_API_KEY`)

### Happy path

| # | Say this | Expect |
|---|---|---|
| 1 | "Add a note titled 'Groceries', body 'milk, eggs', tag shopping" | Note created immediately, no confirmation needed |
| 2 | "What did I write about the API?" | Finds the "API Redesign Notes" note |
| 3 | "Show my meeting notes" | Finds the three "Team Standup" notes |
| 4 | "Delete the note about the old office address" | Agent proposes deletion — **UI:** chat box disables, a warning card with Yes/No buttons appears. **CLI:** a `[CONFIRM]` prompt asking yes/no |
| 5 | Click "✅ Yes, proceed" (or type `yes` in the CLI) after #4 | Note actually deleted — sidebar list updates (UI), agent confirms in its reply |
| 6 | Repeat #4, then click "❌ No, cancel" (or type `no`) | Nothing deleted, agent acknowledges the cancellation, chat box re-enables |

### Persistence

| # | Steps | Expect |
|---|---|---|
| 7 | Add a note, then **restart the app**, then search for it | Note is still there |
| 8 | Delete a note (confirm it), restart the app, search for it | Note is still gone — not reseeded |
| 9 | Delete `notes.db` (or point `NOTES_DB_PATH` at a fresh file), restart | Back to the original 5 seed notes |

### Human-in-the-loop confirmation (the button-based part)

| # | Steps | Expect |
|---|---|---|
| 10 | Ask to delete a note | Chat input becomes disabled/greyed; you cannot send a new free-text message until you click Yes or No |
| 11 | Click "No, cancel" | Chat input re-enables immediately; note untouched |
| 12 | Ask the model to do something that would require deleting **two** notes in one instruction, e.g. "delete both the API note and the old office address note" | If the model bundles both into one turn, you should see ONE confirmation card listing both changes, not two separate prompts |
| 13 | Approve the bundled confirmation from #12 | Both notes are gone |
| 14 | Repeat #12 but click "No" | Neither note is deleted — rejection applies to the whole bundle |
| 15 | Open the sidebar "Pending" state indirectly: while a confirmation is showing, check the "Tool calls & intent" expander on the *previous* turn — it should show the proposed tool call even though it hasn't executed yet |

### Edge cases

| # | Input | Expect |
|---|---|---|
| 16 | "Delete my standup note" (3 notes share that title) | Agent lists all 3 by ID and asks which one — must not guess, and nothing should reach the confirmation stage until a specific ID is chosen |
| 17 | "Delete note 999" | `ERROR: Note #999 does not exist.` — no confirmation prompt appears at all, since the tool call itself errors before reaching `human_review`'s note lookup — check `_describe_change` still handles a missing note gracefully if the ID existed at proposal time but got deleted in another tab before you clicked Yes |
| 18 | Add a note with an empty title | `ERROR: Title cannot be empty.` — no confirmation involved, this path never touches destructive tools |
| 19 | Ask something conversational ("thanks!", "how are you") while nothing is pending | Normal reply, no tool calls, no confirmation UI |
| 20 | Refresh the browser tab mid-confirmation (UI only) | Streamlit's `@st.cache_resource` keeps the same graph/checkpointer alive for the process, so the pending interrupt should still be there after refresh — worth confirming this actually holds, since it's the trickiest part of wiring a pause into a rerun-every-interaction framework like Streamlit |

## Known non-fixed limitations, for context

- Single-user/single-session — one fixed `thread_id`, one shared DB connection per process.
- SQLite with a single connection + a lock is fine for one user, not built for concurrent writers.
- The intent classifier is a heuristic; it's advisory only now (doesn't gate anything), so a misfire just produces a slightly less helpful hint, not an incorrect action.
- No audit log — once a delete is approved, there's no record of who approved it or when.
