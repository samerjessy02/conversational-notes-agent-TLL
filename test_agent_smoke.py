"""
Smoke tests using a scripted fake LLM (no API key needed) plus a couple of
plain unit tests on the database. Run with:  python test_agent_smoke.py

Every test that touches the graph uses NoteDatabase(":memory:") explicitly.
NEVER let a test use the default "notes.db" path -- it's meant to persist
across real runs, so a test writing to it would corrupt state for the next
run.
"""
import os
import tempfile
import uuid
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

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


def seeded_memory_db() -> NoteDatabase:
    db = NoteDatabase(":memory:")
    db.add_note("Team Standup", "Agreed to move standup to Tuesdays at 10 AM.", ["meetings", "team"])
    db.add_note("Team Standup", "Meeting Tuesday", ["meetings", "team"])
    db.add_note("Team Standup", "Meeting Friday", ["meetings", "team"])
    db.add_note("API Redesign Notes", "Need to migrate endpoints from v1 REST to v2 GraphQL.", ["work", "api"])
    db.add_note("Old Office Address", "123 Main St, Suite 400, New York, NY", ["personal", "address"])
    return db


# --------------------------------------------------------------------- #
# Intent classifier (hint-only now)
# --------------------------------------------------------------------- #
def test_intent_classifier():
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

    db2 = make_seeded_db(tmp_path)  # simulate a restart
    notes = db2.search_notes()
    assert len(notes) == 6, "reopening the same file must not lose the 6th note"
    assert any(n["title"] == "Sixth Note" for n in notes)
    print("PASS: notes survive closing and reopening the database file")

    db3 = make_seeded_db(tmp_path)
    assert len(db3.search_notes()) == 6, "reopening a non-empty DB must not re-seed duplicates"
    print("PASS: reopening a non-empty DB does not duplicate the seed notes")

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
# Human-in-the-loop confirmation via interrupt()/Command(resume=...)
# --------------------------------------------------------------------- #
def test_delete_requires_explicit_approval():
    db = seeded_memory_db()
    assert db.get_note(5)["title"] == "Old Office Address"

    script = [
        AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 5})]),
    ]
    llm = ScriptedChatModel(responses=script)
    app, db = build_app(db=db, llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    result = app.invoke({"messages": [HumanMessage(content="delete the old office address note")]}, config=config)
    assert "__interrupt__" in result, "a destructive tool call must pause the graph"
    assert db.get_note(5) is not None, "nothing may be deleted before approval"
    payload = result["__interrupt__"][0].value
    assert payload["changes"][0]["note_id"] == 5
    assert "Old Office Address" in payload["changes"][0]["summary"]
    print(f"PASS: delete_note pauses with a clear payload: {payload['changes'][0]['summary']!r}")

    # Reject
    llm.responses.append(AIMessage(content="Okay, I won't delete it."))
    result2 = app.invoke(Command(resume={"approved": False}), config=config)
    assert "__interrupt__" not in result2
    assert db.get_note(5) is not None, "note must survive a rejected delete"
    print(f"PASS: resuming with approved=False cancels cleanly. Reply: {result2['messages'][-1].content!r}")

    # Re-propose then approve
    llm.responses += [
        AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 5})]),
    ]
    result3 = app.invoke({"messages": [HumanMessage(content="actually delete it")]}, config=config)
    assert "__interrupt__" in result3

    llm.responses.append(AIMessage(content="Done, it's deleted."))
    result4 = app.invoke(Command(resume={"approved": True}), config=config)
    assert "__interrupt__" not in result4
    assert db.get_note(5) is None, "note must actually be deleted after explicit approval"
    print(f"PASS: resuming with approved=True actually deletes. Reply: {result4['messages'][-1].content!r}")


def test_modify_requires_explicit_approval():
    db = seeded_memory_db()
    script = [
        AIMessage(content="", tool_calls=[tc("modify_note", {"note_id": 1, "title": "New Title"})]),
    ]
    llm = ScriptedChatModel(responses=script)
    app, db = build_app(db=db, llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t2"}}

    result = app.invoke({"messages": [HumanMessage(content="rename note 1 to New Title")]}, config=config)
    assert "__interrupt__" in result
    assert db.get_note(1)["title"] != "New Title", "must not mutate before approval"

    llm.responses.append(AIMessage(content="Renamed it."))
    result2 = app.invoke(Command(resume={"approved": True}), config=config)
    assert db.get_note(1)["title"] == "New Title", "must mutate after approval"
    print("PASS: modify_note also gates on explicit approval before writing")


def test_bundled_multi_change_turn_all_or_nothing():
    """If the model proposes two destructive actions in the SAME turn, they
    should be bundled into one confirmation payload and resolved together."""
    db = seeded_memory_db()
    script = [
        AIMessage(
            content="",
            tool_calls=[tc("delete_note", {"note_id": 1}), tc("delete_note", {"note_id": 2})],
        ),
    ]
    llm = ScriptedChatModel(responses=script)
    app, db = build_app(db=db, llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t3"}}

    result = app.invoke({"messages": [HumanMessage(content="delete the first two standup notes")]}, config=config)
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert len(payload["changes"]) == 2, "both proposed deletions must appear in one payload"
    print(f"PASS: two destructive calls in one turn bundle into one confirmation ({len(payload['changes'])} changes)")

    llm.responses.append(AIMessage(content="Both deleted."))
    result2 = app.invoke(Command(resume={"approved": True}), config=config)
    assert db.get_note(1) is None and db.get_note(2) is None, "approving once must apply BOTH changes"
    print("PASS: a single approval applies every bundled change")


def test_rejecting_bundled_changes_applies_neither():
    db = seeded_memory_db()
    script = [
        AIMessage(
            content="",
            tool_calls=[tc("delete_note", {"note_id": 1}), tc("delete_note", {"note_id": 2})],
        ),
    ]
    llm = ScriptedChatModel(responses=script)
    app, db = build_app(db=db, llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t4"}}

    app.invoke({"messages": [HumanMessage(content="delete the first two standup notes")]}, config=config)
    llm.responses.append(AIMessage(content="Okay, cancelled both."))
    result = app.invoke(Command(resume={"approved": False}), config=config)
    assert db.get_note(1) is not None and db.get_note(2) is not None, "rejecting once must cancel BOTH changes"
    print("PASS: a single rejection cancels every bundled change, neither is applied")


def test_sequential_proposals_each_get_their_own_interrupt():
    """Reproduces a real observed sequence: the model proposes deleting note
    A, gets rejected, and on its OWN NEXT TURN (not bundled -- a separate
    AIMessage) decides to propose deleting note B too. Note B's deletion
    must require its own independent approval -- rejecting note A must
    never implicitly grant or skip approval for note B."""
    db = seeded_memory_db()
    script = [
        AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 4})]),
        AIMessage(
            content="Cancelled note #4. I'll delete note #5 as well.",
            tool_calls=[tc("delete_note", {"note_id": 5})],
        ),
        AIMessage(content="Deleted note #5. Note #4 is untouched."),
    ]
    llm = ScriptedChatModel(responses=script)
    app, db = build_app(db=db, llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t5"}}

    r1 = app.invoke({"messages": [HumanMessage(content="delete the API note and the office note")]}, config=config)
    assert "__interrupt__" in r1
    assert r1["__interrupt__"][0].value["changes"][0]["note_id"] == 4

    r2 = app.invoke(Command(resume={"approved": False}), config=config)
    assert "__interrupt__" in r2, "the model's follow-up proposal must trigger its OWN interrupt"
    assert r2["__interrupt__"][0].value["changes"][0]["note_id"] == 5
    assert db.get_note(4) is not None, "rejecting note 4 must not affect it either way beyond cancelling"
    assert db.get_note(5) is not None, "note 5 must NOT be deleted just because it was mentioned in the same reply"
    print("PASS: a follow-up proposal after a rejection gets its own independent interrupt")

    r3 = app.invoke(Command(resume={"approved": True}), config=config)
    assert "__interrupt__" not in r3
    assert db.get_note(4) is not None, "note 4 was rejected earlier and must still be untouched"
    assert db.get_note(5) is None, "note 5 was separately approved and should now be deleted"
    print("PASS: approving the second, independent interrupt only affects note 5, not note 4")


def test_stubborn_model_repeating_rejected_proposal_is_hard_stopped():
    """Reproduces the reported bug: user says 'delete both X and Y', rejects
    the first proposal, and the model -- instead of accepting the rejection
    or moving on -- immediately re-proposes the EXACT SAME deletion again.
    Before the circuit breaker, this could force the human to keep clicking
    'No' indefinitely. Now: the second identical proposal must be
    auto-cancelled and the turn must end WITHOUT a second interrupt --
    i.e. app.invoke() returns cleanly with no "__interrupt__" the moment the
    model retries a rejected target, no matter how many more times it tries."""
    db = seeded_memory_db()
    script = [
        AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 4})]),
        # Model stubbornly re-proposes the SAME note right after rejection,
        # instead of accepting it or trying something else.
        AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 4})]),
    ]
    llm = ScriptedChatModel(responses=script)
    app, db = build_app(db=db, llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t6"}}

    r1 = app.invoke({"messages": [HumanMessage(content="delete the API note")]}, config=config)
    assert "__interrupt__" in r1

    r2 = app.invoke(Command(resume={"approved": False}), config=config)
    assert "__interrupt__" not in r2, (
        "the circuit breaker must auto-cancel a repeat proposal and END the turn "
        "instead of interrupting AGAIN and making the human reject a second time"
    )
    assert llm.idx == len(script), "the LLM must NOT be called a third time -- the breaker ends the turn directly"
    assert db.get_note(4) is not None, "the note must still be untouched"
    last_content = r2["messages"][-1].content
    assert "won't ask again" in last_content or "already declined" in last_content
    print(f"PASS: a repeated identical proposal is auto-cancelled and the turn ends immediately. Reply: {last_content!r}")

    # A genuinely NEW user message must be able to try again -- the breaker
    # is scoped to "this turn", not permanent.
    llm.responses.append(AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 4})]))
    r3 = app.invoke({"messages": [HumanMessage(content="ok actually delete the API note now")]}, config=config)
    assert "__interrupt__" in r3, "a fresh user message must be able to propose the same note again"
    print("PASS: a new user message resets the breaker and can propose the same note again")


if __name__ == "__main__":
    test_intent_classifier()
    test_persistence_across_reopen()
    test_delete_persists_across_reopen()
    test_delete_requires_explicit_approval()
    test_modify_requires_explicit_approval()
    test_bundled_multi_change_turn_all_or_nothing()
    test_rejecting_bundled_changes_applies_neither()
    test_sequential_proposals_each_get_their_own_interrupt()
    test_stubborn_model_repeating_rejected_proposal_is_hard_stopped()
    print("\nALL SMOKE TESTS PASSED")
