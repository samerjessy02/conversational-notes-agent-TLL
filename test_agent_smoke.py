import uuid
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from agent import build_app, make_seeded_db, classify_intent


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


def test_graph_flow():
    db = make_seeded_db()
    script = [
        # "delete note 5" -> tool call proposes deletion (does NOT delete)
        AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 5})]),
        AIMessage(content="Note #5 ('Old Office Address') is staged for deletion. Confirm?"),
    ]
    llm = ScriptedChatModel(responses=script)
    app, db = build_app(db=db, llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    out = app.invoke({"messages": [HumanMessage(content="delete the old office address note")]}, config=config)
    assert db.get_note(5) is not None, "tool must NOT delete on the proposing call"
    assert out["pending_confirmation"]["tool"] == "delete_note"
    assert out["pending_confirmation"]["note_id"] == 5
    print("PASS: delete_note proposes without mutating the DB")

    # User says "no" -- deterministic short-circuit, LLM is never called again
    idx_before = llm.idx
    out2 = app.invoke({"messages": [HumanMessage(content="no")]}, config=config)
    assert llm.idx == idx_before, "confirm_no must bypass the LLM entirely"
    assert out2["pending_confirmation"] is None
    assert db.get_note(5) is not None, "note must survive a rejected delete"
    print(f"PASS: 'no' cancels deterministically without calling the LLM. Reply: {out2['messages'][-1].content!r}")

    # Re-propose then approve
    llm.responses.append(AIMessage(content="", tool_calls=[tc("delete_note", {"note_id": 5})]))
    llm.responses.append(AIMessage(content="Note #5 is staged for deletion again. Confirm?"))
    out3 = app.invoke({"messages": [HumanMessage(content="actually delete the old office address note")]}, config=config)
    assert out3["pending_confirmation"]["note_id"] == 5

    idx_before = llm.idx
    out4 = app.invoke({"messages": [HumanMessage(content="yes")]}, config=config)
    assert llm.idx == idx_before, "confirm_yes must bypass the LLM entirely"
    assert db.get_note(5) is None, "note must actually be deleted after explicit yes"
    assert out4["pending_confirmation"] is None
    print(f"PASS: 'yes' actually deletes deterministically. Reply: {out4['messages'][-1].content!r}")

    # Mismatched confirmation attempt: LLM hallucinating confirmed args on an
    # unrelated note should be impossible now since there's no `confirmed`
    # field at all -- modify_note always just proposes.
    llm.responses.append(AIMessage(content="", tool_calls=[tc("modify_note", {"note_id": 1, "title": "New Title"})]))
    llm.responses.append(AIMessage(content="Note #1 is staged to be renamed. Confirm?"))
    out5 = app.invoke({"messages": [HumanMessage(content="rename note 1 to 'New Title'")]}, config=config)
    assert db.get_note(1)["title"] != "New Title", "modify_note must never mutate on the proposing call"
    assert out5["pending_confirmation"]["note_id"] == 1
    print("PASS: modify_note also only proposes, never mutates directly")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    test_intent_classifier()
    test_graph_flow()
