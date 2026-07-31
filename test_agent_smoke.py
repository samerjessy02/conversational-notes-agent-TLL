"""
Smoke tests using a scripted fake LLM (no API key needed) plus a couple of
plain unit tests on the database. Run with:  python test_agent_smoke.py

Every test that touches the graph uses db_path=":memory:" explicitly.
NEVER let a test use the default "notes.db" path -- it's meant to persist
across real runs, so a test writing to it would corrupt state for the next
run (this bit us once already: a delete during a test would silently
survive to the next `python test_agent_smoke.py` invocation).
"""
import os
import tempfile
import uuid
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from agent import NoteDatabase, build_app, classify_intent, make_seeded_db


class ScriptedChatModel(BaseChatModel):
    responses: List[AIMessage] = []
    idx: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.responses[self.idx]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self


def tc(name, args):
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}


# --------------------------------------------------------------------- #
# Intent classifier
# --------------------------------------------------------------------- #
def test_intent_classifier():
    assert classify_intent("yes") == "confirm_yes"
    assert classify_intent("Yeah go ahead") == "confirm_yes"
    assert classify_intent("no") == "confirm_no"
    assert classify_intent("nope, cancel that") == "confirm_no"
    assert classify_intent("delete the office note") == "delete"
    assert classify_intent("can you update my standup note") == "modify"
    assert classify_intent("add a note about groceries") == "create"
    assert classify_intent("what did I write about the API last week") == "search"
    assert classify_intent("how's it going") == "chitchat"
    print("PASS: classify_intent covers the basic cases")


# --------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------- #
def test_persistence_across_reopen():
    tmp_path = os.path.join(tempfile.mkdtemp(), "test_notes.db")

    db1 = make_seeded_db(tmp_path)
    assert len(db1.search_notes()) == 5, "fresh file should get the 5 seed notes"
    db1.add_note("Sixth Note", "added after seeding", ["scratch"])
    assert len(db1.search_notes()) == 6

    # Simulate a restart: open a brand new NoteDatabase pointed at the same file
    db2 = make_seeded_db(tmp_path)
    notes = db2.search_notes()
    assert len(notes) == 6, "reopening the same file must not lose the 6th note"
    assert any(n["title"] == "Sixth Note" for n in notes)
    print("PASS: notes survive closing and reopening the database file")

    # And seeding must NOT run again on a non-empty file
    db3 = make_seeded_db(tmp_path)
    assert len(db3.search_notes()) == 6, "reopening a non-empty DB must not re-seed duplicates"
    print("PASS: reopening a non-empty DB does not duplicate the seed notes")

    # A genuinely fresh path, on the other hand, should get freshly seeded
    tmp_path2 = os.path.join(tempfile.mkdtemp(), "other_notes.db")
    db4 = make_seeded_db(tmp_path2)
    assert len(db4.search_notes()) == 5
    print("PASS: a fresh path still gets seeded normally")


def test_delete_persists_across_reopen():
    tmp_path = os.path.join(tempfile.mkdtemp(), "test_notes2.db")
    db1 = make_seeded_db(tmp_path)
    db1.delete_note(5)
    assert db1.get_note(5) is None

    db2 = make_seeded_db(tmp_path)  # simulate restart
    assert db2.get_note(5) is None, "a deletion made before restart must still be gone after restart"
    assert len(db2.search_notes()) == 4, "and it must not have been re-seeded back in"
    print("PASS: a deletion survives a restart and isn't re-seeded")


# --------------------------------------------------------------------- #
# Confirmation queue + graph flow
# --------------------------------------------------------------------- #
def test_graph_flow_basic():
    db = NoteDatabase(":memory:")
    if db.is_empty():
        db.add_note("Team Standup", "Agreed to move standup to Tuesdays at 10 AM.", ["meetings", "team"])
        db.add_note("Team Standup", "Meeting Tuesday", ["meetings", "team"])
        db.add_note("Team Standup", "Meeting Friday", ["meetings", "team"])
        db.add_note("API Redesign Notes", "Need to migrate endpoints from v1 REST to v2 GraphQL.", ["work", "api"])
        db.add_note("Old Office Address", "123 Main St, Suite 400, New York, NY", ["personal", "address"])
    assert db.get_note(5)["title"] == "Old Office Address"

    script = [
        AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 5})]),
        AIMessage(content="Note #5 is staged for deletion. Confirm?"),
    ]
    llm = ScriptedChatModel(responses=script)
    app, db = build_app(db=db, llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    out = app.invoke({"messages": [HumanMessage(content="delete the old office address note")]}, config=config)
    assert db.get_note(5) is not None, "tool must NOT delete on the proposing call"
    assert len(out["pending_confirmations"]) == 1
    assert out["pending_confirmations"][0]["tool"] == "delete_note"
    assert out["pending_confirmations"][0]["note_id"] == 5
    print("PASS: delete_note proposes without mutating the DB")

    idx_before = llm.idx
    out2 = app.invoke({"messages": [HumanMessage(content="no")]}, config=config)
    assert llm.idx == idx_before, "confirm_no must bypass the LLM entirely"
    assert out2["pending_confirmations"] == []
    assert db.get_note(5) is not None
    print(f"PASS: 'no' cancels deterministically without calling the LLM. Reply: {out2['messages'][-1].content!r}")

    llm.responses += [
        AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 5})]),
        AIMessage(content="Note #5 is staged for deletion again. Confirm?"),
    ]
    out3 = app.invoke({"messages": [HumanMessage(content="actually delete it")]}, config=config)
    assert len(out3["pending_confirmations"]) == 1

    idx_before = llm.idx
    out4 = app.invoke({"messages": [HumanMessage(content="yes")]}, config=config)
    assert llm.idx == idx_before, "confirm_yes must bypass the LLM entirely"
    assert db.get_note(5) is None, "note must actually be deleted after explicit yes"
    assert out4["pending_confirmations"] == []
    print(f"PASS: 'yes' actually deletes deterministically. Reply: {out4['messages'][-1].content!r}")


def test_confirmation_queue_stacking():
    """Two destructive proposals stage before either is confirmed. A bare
    'yes' should resolve the most recently proposed one first, and the
    first one should still be waiting afterward -- not silently dropped."""
    db = NoteDatabase(":memory:")
    db.add_note("Note A", "body A", [])
    db.add_note("Note B", "body B", [])

    script = [
        AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 1})]),
        AIMessage(content="Note #1 staged for deletion."),
        AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 2})]),
        AIMessage(content="Note #2 staged for deletion."),
    ]
    llm = ScriptedChatModel(responses=script)
    app, db = build_app(db=db, llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t2"}}

    app.invoke({"messages": [HumanMessage(content="delete note A")]}, config=config)
    out = app.invoke({"messages": [HumanMessage(content="delete note B too")]}, config=config)
    assert len(out["pending_confirmations"]) == 2, "both proposals must be queued, not overwritten"
    assert [p["note_id"] for p in out["pending_confirmations"]] == [1, 2]
    print("PASS: a second proposal queues alongside the first instead of overwriting it")

    out2 = app.invoke({"messages": [HumanMessage(content="yes")]}, config=config)
    assert db.get_note(2) is None, "the MOST RECENTLY proposed item (note B) resolves first"
    assert db.get_note(1) is not None, "the earlier proposal (note A) must still be intact"
    assert len(out2["pending_confirmations"]) == 1
    assert "pending action" in out2["messages"][-1].content
    print(f"PASS: 'yes' resolves the most recent proposal and flags the remaining one. Reply: {out2['messages'][-1].content!r}")

    out3 = app.invoke({"messages": [HumanMessage(content="yes")]}, config=config)
    assert db.get_note(1) is None
    assert out3["pending_confirmations"] == []
    print("PASS: a second 'yes' resolves the remaining queued item")


def test_restage_same_note_replaces_not_duplicates():
    """Proposing a change to the same note twice before confirming should
    replace the pending entry, not create two queued confirmations for the
    same note."""
    db = NoteDatabase(":memory:")
    db.add_note("Note A", "original body", [])

    script = [
        AIMessage(content="", tool_calls=[tc("modify_note", {"note_id": 1, "title": "First Rename"})]),
        AIMessage(content="Staged."),
        AIMessage(content="", tool_calls=[tc("modify_note", {"note_id": 1, "title": "Second Rename"})]),
        AIMessage(content="Staged again."),
    ]
    llm = ScriptedChatModel(responses=script)
    app, db = build_app(db=db, llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t3"}}

    app.invoke({"messages": [HumanMessage(content="rename note 1 to First Rename")]}, config=config)
    out = app.invoke({"messages": [HumanMessage(content="actually rename it to Second Rename instead")]}, config=config)
    assert len(out["pending_confirmations"]) == 1, "re-proposing the same note must replace, not stack"
    assert out["pending_confirmations"][0]["title"] == "Second Rename"
    print("PASS: re-proposing a change to the same note replaces the pending entry")

    out2 = app.invoke({"messages": [HumanMessage(content="yes")]}, config=config)
    assert db.get_note(1)["title"] == "Second Rename", "the LATEST proposal must be the one applied"
    print("PASS: confirming applies the latest staged version, not a stale one")


if __name__ == "__main__":
    test_intent_classifier()
    test_persistence_across_reopen()
    test_delete_persists_across_reopen()
    test_graph_flow_basic()
    test_confirmation_queue_stacking()
    test_restage_same_note_replaces_not_duplicates()
    print("\nALL SMOKE TESTS PASSED")
