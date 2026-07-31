# Test Cases

Two kinds of tests here. The automated ones (`test_agent_smoke.py`) check
the mechanics — state transitions, the DB, the confirmation queue — using a
scripted fake LLM, so they're exact and repeatable. The manual ones exercise
the real model's tool-calling and phrasing judgement, which by nature varies
a bit run to run — treat "expected" there as "what a reasonable model should
do," not a byte-exact string match.

## Automated (`python test_agent_smoke.py`)

| Test | What it proves |
|---|---|
| `test_intent_classifier` | Classifier labels basic create/search/modify/delete/yes/no/chitchat messages correctly |
| `test_persistence_across_reopen` | Notes added in one `NoteDatabase` instance are visible after closing and reopening a new instance on the same file path; reopening a non-empty file does not re-seed |
| `test_delete_persists_across_reopen` | A deletion made before a simulated restart is still gone after |
| `test_graph_flow_basic` | `delete_note` never touches the DB on the proposing call; `"no"` cancels without calling the LLM; `"yes"` deletes without calling the LLM |
| `test_confirmation_queue_stacking` | Two proposals before either is confirmed both get queued (not overwritten); a bare `"yes"` resolves the most recent one first; the older one is still there afterward |
| `test_restage_same_note_replaces_not_duplicates` | Proposing a second change to the same note before confirming replaces the pending entry instead of queuing a duplicate; confirming applies the latest version |

Run it any time with `python test_agent_smoke.py` (or `uv run python
test_agent_smoke.py`) — no API key required, it stubs the LLM.

## Manual (run against the real CLI or UI, with a real `GROQ_API_KEY`)

### Happy path

| # | Say this | Expect |
|---|---|---|
| 1 | "Add a note titled 'Groceries', body 'milk, eggs', tag shopping" | Note created immediately, no confirmation asked |
| 2 | "What did I write about the API?" | Finds the "API Redesign Notes" note |
| 3 | "Show my meeting notes" | Finds the three "Team Standup" notes (tag or keyword match) |
| 4 | "Delete the note about the old office address" | Agent searches, finds it, proposes deletion, asks to confirm |
| 5 | "yes" (after #4) | Note actually deleted — check the sidebar list updates |
| 6 | Repeat #4, then say "no" | Nothing deleted, agent acknowledges the cancellation |

### Persistence (the new part)

| # | Steps | Expect |
|---|---|---|
| 7 | Add a note, then **restart the app** (stop and rerun `main.py` / `streamlit run ui.py`), then search for it | Note is still there |
| 8 | Delete a note, confirm with "yes", restart the app, search for it | Note is still gone — not reseeded |
| 9 | Delete `notes.db` (or point `NOTES_DB_PATH` at a fresh file), restart | Back to the original 5 seed notes |
| 10 | Set `NOTES_DB_PATH=other.db`, add a note, then restart pointing at the default `notes.db` again | The note from `other.db` should NOT appear — confirms notes are scoped per file, not global |

### Confirmation queue (the other new part)

| # | Steps | Expect |
|---|---|---|
| 11 | "Delete note 1" then, before answering, "also delete note 2" | Agent proposes the second deletion too, without silently dropping the first — check the UI's "Pending confirmations" panel shows 2 |
| 12 | Then say "yes" | Note 2 (the most recently proposed) is deleted; note 1 is untouched; reply mentions there's still a pending action for note 1 |
| 13 | Say "yes" again | Note 1 is now deleted too |
| 14 | Propose renaming note 3, then before confirming, ask to rename it to something else instead | Only one pending action for note 3 (the latest title), not two |
| 15 | Propose a delete, then send an unrelated chitchat message ("thanks!") without saying yes/no | Pending confirmation should still be there afterward — an unrelated message shouldn't clear or resolve it |

### Edge cases

| # | Input | Expect |
|---|---|---|
| 16 | "Delete my standup note" (3 notes share that title) | Agent lists all 3 by ID and asks which one — must not guess |
| 17 | "yes" with nothing pending | Falls through to a normal conversational reply, doesn't error |
| 18 | "Delete note 999" | `ERROR: Note #999 does not exist.` — nothing gets queued for confirmation |
| 19 | "yeah, go ahead" / "nah, don't" | Should still resolve as yes/no (covered by `test_intent_classifier`) |
| 20 | "no, delete note 3 instead" | Should NOT be read as a bare rejection — the classifier only treats yes/no words as a bare confirmation for short (≤6 word) messages, so this falls through to a `delete` intent |
| 21 | Add a note with an empty title | `ERROR: Title cannot be empty.` |
| 22 | Ask "what do I have tagged nonexistent-tag" | `NO_MATCHES` — clean empty-result message, not an error |
| 23 | Search with only a relative-time phrase, e.g. "what did I add yesterday" | The temporal-word stripper removes "yesterday", so the query becomes empty — verify this doesn't crash and behaves sensibly (likely returns everything or nothing depending on how the model then calls the tool; worth eyeballing, not a hard assertion) |

## Known non-fixed limitations, for context

- Single shared conversation thread in the UI (`thread_id="ui-session"`) — not multi-user.
- SQLite with a single connection + a lock is fine for one user clicking around; it is not built for concurrent writers.
- The confirmation queue has no expiry — a proposal from an hour ago is still there waiting for a "yes" if nothing else cleared it.
