"""
Core agent: database, tools, intent classifier, and the LangGraph workflow.

Both main.py (CLI) and ui.py (Streamlit) import build_app() from here so
there's exactly one implementation instead of two copies drifting apart.

WHAT CHANGED FROM THE ORIGINAL FILE, AND WHY
----------------------------------------------------------------------------
1. Intent classifier: classify_intent() is a small, fast, deterministic
   (regex/keyword) classifier that tags every user turn with one of
   create / search / modify / delete / confirm_yes / confirm_no / chitchat.
   It's used two ways:
     - As a hint injected into the system prompt (e.g. "this looks like a
       delete request") so the LLM is nudged toward the right tool.
     - To deterministically short-circuit "yes"/"no" replies straight to
       execute_confirmation_node (see #2) instead of asking the LLM to
       decide what a bare "yes" means.
   This is intentionally rule-based rather than a second LLM call: it's
   free, has zero latency, and its output is directly inspectable in the
   UI's trace panel for debugging misroutes.

2. Confirmation is no longer a boolean the LLM is trusted to set correctly.
   In the original code, modify_note_tool/delete_note_tool trusted a
   `confirmed: bool` argument the model itself supplied -- nothing stopped
   the model from setting confirmed=True on the first call, or from
   confirming note #2 when the user actually approved changes to note #1
   (a realistic failure mode with smaller/faster tool-calling models).

   Now: modify_note_tool and delete_note_tool NEVER mutate the database.
   They only look up the note and *propose* a change, writing the proposal
   into graph state (`pending_confirmation`) via a `Command` update. The
   actual mutation only happens in execute_confirmation_node, which the
   graph reaches only when classify_intent() has deterministically read
   "yes"/"no" from the user's own words on the very next turn. The LLM is
   removed from the confirmation-authority path entirely.

3. Bug fix: the original REQUIRES_CONFIRMATION change-description used
   `if title:` / `if tags:` (truthy checks), so explicitly clearing a field
   to "" or [] silently wouldn't show up as a change. Fixed to `is not None`.

4. Startup check for GROQ_API_KEY with a clear error instead of a confusing
   stack trace on the first LLM call.
"""
import datetime
import os
import re
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

load_dotenv()


# =====================================================================
# 1. DATABASE LAYER
# =====================================================================
class NoteDatabase:
    """In-memory note storage simulating a backend database."""

    def __init__(self):
        self.notes: Dict[int, Dict[str, Any]] = {}
        self.next_id = 1

    def add_note(self, title: str, body: str, tags: List[str]) -> Dict[str, Any]:
        note = {
            "id": self.next_id,
            "title": title,
            "body": body,
            "tags": [t.lower().strip() for t in tags],
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.notes[self.next_id] = note
        self.next_id += 1
        return note

    def search_notes(self, query: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        results = list(self.notes.values())

        if tag:
            tag = tag.lower().strip()
            results = [n for n in results if tag in n["tags"]]

        if query:
            query = query.lower()
            query = re.sub(r"\b(last|this|today|yesterday|week|month|year)\b", "", query)
            words = [w.strip() for w in query.split() if w.strip()]
            filtered = []
            for note in results:
                text = (note["title"] + " " + note["body"] + " " + " ".join(note["tags"])).lower()
                if all(word in text for word in words):
                    filtered.append(note)
            results = filtered

        return results

    def get_note(self, note_id: int) -> Optional[Dict[str, Any]]:
        return self.notes.get(note_id)

    def modify_note(
        self, note_id: int, title: Optional[str] = None, body: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        note = self.notes.get(note_id)
        if not note:
            return None
        if title is not None:
            note["title"] = title
        if body is not None:
            note["body"] = body
        if tags is not None:
            note["tags"] = [t.lower().strip() for t in tags]
        return note

    def delete_note(self, note_id: int) -> bool:
        if note_id in self.notes:
            del self.notes[note_id]
            return True
        return False


def make_seeded_db() -> NoteDatabase:
    db = NoteDatabase()
    db.add_note("Team Standup", "Agreed to move standup to Tuesdays at 10 AM.", ["meetings", "team"])
    db.add_note("Team Standup", "Meeting Tuesday", ["meetings", "team"])
    db.add_note("Team Standup", "Meeting Friday", ["meetings", "team"])
    db.add_note("API Redesign Notes", "Need to migrate endpoints from v1 REST to v2 GraphQL.", ["work", "api"])
    db.add_note("Old Office Address", "123 Main St, Suite 400, New York, NY", ["personal", "address"])
    return db


# =====================================================================
# 2. INTENT CLASSIFIER
# =====================================================================
_YES_RE = re.compile(r"\b(yes|yeah|yep|yup|sure|confirm(ed)?|correct|go ahead|do it|okay|ok)\b", re.I)
_NO_RE = re.compile(r"\b(no|nope|nah|cancel|don'?t|do not|stop|never ?mind)\b", re.I)
_DELETE_RE = re.compile(r"\b(delete|remove|erase|trash|get rid of)\b", re.I)
_MODIFY_RE = re.compile(r"\b(update|change|edit|modify|rename|correct|fix)\b", re.I)
_CREATE_RE = re.compile(r"\b(add|create|new note|jot down|write down|save a note|note that|remember that)\b", re.I)
_SEARCH_RE = re.compile(r"\b(find|search|show|list|display|what did i|look ?up|recall|any notes)\b", re.I)

Intent = Literal["confirm_yes", "confirm_no", "delete", "modify", "create", "search", "chitchat"]


def classify_intent(text: str) -> Intent:
    """Cheap, deterministic, regex-based intent tag for a user message.

    This is intentionally simple: it doesn't need to be perfect, since for
    create/search/modify/delete it's only used as a *hint* the LLM can
    override. For confirm_yes/confirm_no it IS load-bearing (see
    route_after_classify below), which is why those two patterns are
    deliberately narrow (whole-message match) rather than a keyword search
    anywhere in the text -- we don't want "no thanks, but can you delete
    note 3" to be misread as a plain rejection.
    """
    t = text.strip()
    # Only treat yes/no words as a bare confirmation for short replies, so a
    # full sentence like "sure, but first delete note 3" or "no, delete note
    # 3 instead" falls through to intent detection below rather than being
    # misread as a plain confirmation.
    if len(t.split()) <= 6:
        if _NO_RE.search(t):
            return "confirm_no"
        if _YES_RE.search(t):
            return "confirm_yes"
    if _DELETE_RE.search(t):
        return "delete"
    if _MODIFY_RE.search(t):
        return "modify"
    if _CREATE_RE.search(t):
        return "create"
    if _SEARCH_RE.search(t):
        return "search"
    return "chitchat"


_INTENT_HINTS = {
    "delete": "The user's message looks like a DELETE request. Search first if you don't already "
    "know the note's ID, then call delete_note -- do not ask the user to confirm yourself, "
    "the system handles that automatically after your tool call.",
    "modify": "The user's message looks like a MODIFY/UPDATE request. Search first if you don't "
    "already know the note's ID, then call modify_note with just the fields that should change.",
    "create": "The user's message looks like a CREATE request. Call add_note directly.",
    "search": "The user's message looks like a SEARCH request. Call search_notes with concise "
    "keywords (no dates/relative-time words).",
}


# =====================================================================
# 3. STATE
# =====================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    pending_confirmation: Optional[Dict[str, Any]]
    last_intent: Optional[str]


# =====================================================================
# 4. TOOLS
# =====================================================================
class AddNoteInput(BaseModel):
    title: str = Field(description="A concise, descriptive title for the note.")
    body: str = Field(description="The main text body of the note.")
    tags: Optional[List[str]] = Field(default=[], description="Keywords/categories for filtering.")


class SearchNotesInput(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description="Main keyword or topic to search for (e.g. 'API', 'standup', 'office'). "
        "Do NOT include dates or relative-time phrases like 'last week'.",
    )
    tag: Optional[str] = Field(
        default=None,
        description="A single tag to filter by (e.g. 'meetings', 'work'). Leave empty unless the "
        "user explicitly refers to a tag.",
    )


class ModifyNoteInput(BaseModel):
    note_id: int = Field(description="The unique integer ID of the note to modify.")
    title: Optional[str] = Field(default=None, description="Updated title text, if changing.")
    body: Optional[str] = Field(default=None, description="Updated body content, if changing.")
    tags: Optional[List[str]] = Field(default=None, description="Updated tag list, if changing.")


class DeleteNoteInput(BaseModel):
    note_id: int = Field(description="The unique integer ID of the note to remove.")


def build_tools(db: NoteDatabase):
    @tool("add_note", args_schema=AddNoteInput)
    def add_note_tool(title: str, body: str, tags: Optional[List[str]] = None) -> str:
        """Create and store a new note in the system. Does not require confirmation."""
        if not title or not title.strip():
            return "ERROR: Title cannot be empty."
        note = db.add_note(title, body, tags or [])
        return f"SUCCESS: Note #{note['id']} created. Title: '{note['title']}'."

    @tool("search_notes", args_schema=SearchNotesInput)
    def search_notes_tool(query: Optional[str] = None, tag: Optional[str] = None) -> str:
        """Retrieve existing notes by keyword, topic, or tag."""
        results = db.search_notes(query=query, tag=tag)
        if not results:
            return "NO_MATCHES: No notes found matching your criteria."
        output = [f"Found {len(results)} matching note(s):"]
        for n in results:
            output.append(f"- ID #{n['id']}: [{n['title']}] Body: '{n['body']}' (Tags: {', '.join(n['tags'])})")
        return "\n".join(output)

    @tool("modify_note", args_schema=ModifyNoteInput)
    def modify_note_tool(
        note_id: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = None,
    ) -> Any:
        """Propose a modification to an existing note's title, body, or tags.
        This does NOT change anything yet -- it stages the change and the
        system will ask the user to confirm on the next turn."""
        note = db.get_note(note_id)
        if not note:
            return f"ERROR: Note #{note_id} does not exist."

        changes = []
        if title is not None:
            changes.append(f"Title -> '{title}'")
        if body is not None:
            changes.append(f"Body -> '{body}'")
        if tags is not None:
            changes.append(f"Tags -> {tags}")
        change_desc = ", ".join(changes) if changes else "no changes specified"

        msg = (
            f"REQUIRES_CONFIRMATION: Target Note #{note_id} ('{note['title']}'). "
            f"Proposed changes: [{change_desc}]. Ask the user to reply yes/no to confirm."
        )
        return Command(
            update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
                "pending_confirmation": {
                    "tool": "modify_note",
                    "note_id": note_id,
                    "title": title,
                    "body": body,
                    "tags": tags,
                    "summary": change_desc,
                },
            }
        )

    @tool("delete_note", args_schema=DeleteNoteInput)
    def delete_note_tool(note_id: int, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Any:
        """Propose deleting a note permanently. This does NOT delete anything
        yet -- it stages the deletion and the system will ask the user to
        confirm on the next turn."""
        note = db.get_note(note_id)
        if not note:
            return f"ERROR: Note #{note_id} does not exist."

        msg = (
            f"REQUIRES_CONFIRMATION: Delete Note #{note_id} ('{note['title']}') permanently? "
            "Ask the user to reply yes/no to confirm."
        )
        return Command(
            update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
                "pending_confirmation": {
                    "tool": "delete_note",
                    "note_id": note_id,
                    "summary": f"delete note #{note_id} ('{note['title']}')",
                },
            }
        )

    return [add_note_tool, search_notes_tool, modify_note_tool, delete_note_tool]


# =====================================================================
# 5. SYSTEM PROMPT
# =====================================================================
SYSTEM_PROMPT = """You are a conversational note-taking assistant.

SEARCH RULES
Extract only the important keyword(s) for search_notes -- never include dates or
relative-time phrases like "last week" or "yesterday". Use `tag` only when the
user explicitly references a tag by name.

MODIFICATION / DELETION RULES
Always search first if you don't already know the note's ID. Never guess a note.
If multiple notes match, list them (ID + title) and ask the user which one they mean.
Call modify_note / delete_note with the change you intend to make -- you do not
need to ask for confirmation yourself or track any "confirmed" flag; the system
automatically pauses after your tool call, asks the user yes/no, and only applies
the change if they say yes. If you see a ToolMessage cancelling an action, just
acknowledge it.
"""


# =====================================================================
# 6. GRAPH
# =====================================================================
def build_app(db: Optional[NoteDatabase] = None, checkpointer=None, llm=None):
    """Build the compiled graph. `db`, `checkpointer`, and `llm` are all
    injectable so the CLI, the UI, and tests can each supply their own
    (e.g. tests can pass a fake llm instead of hitting the real Groq API).
    Returns (app, db).
    """
    if db is None:
        db = make_seeded_db()

    if llm is None:
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to a .env file or export it before running."
            )
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    tools = build_tools(db)
    llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

    def classify_intent_node(state: AgentState) -> Dict[str, Any]:
        last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        intent = classify_intent(last_human.content) if last_human else "chitchat"
        return {"last_intent": intent}

    def route_after_classify(state: AgentState) -> Literal["execute_confirmation", "agent"]:
        if state.get("last_intent") in ("confirm_yes", "confirm_no") and state.get("pending_confirmation"):
            return "execute_confirmation"
        return "agent"

    def execute_confirmation_node(state: AgentState) -> Dict[str, Any]:
        """The ONLY place a modify/delete is actually applied to the database.
        Reached only via a deterministic yes/no classification, never via the
        LLM deciding on its own that something is confirmed."""
        pending = state["pending_confirmation"]
        if state["last_intent"] == "confirm_no":
            reply = f"Okay, cancelled -- I won't {pending['summary']}."
            return {"messages": [AIMessage(content=reply)], "pending_confirmation": None}

        if pending["tool"] == "delete_note":
            note = db.get_note(pending["note_id"])
            if not note:
                reply = f"Note #{pending['note_id']} no longer exists (already deleted?)."
            else:
                db.delete_note(pending["note_id"])
                reply = f"Deleted note #{pending['note_id']} ('{note['title']}')."
        elif pending["tool"] == "modify_note":
            updated = db.modify_note(
                pending["note_id"], title=pending.get("title"), body=pending.get("body"), tags=pending.get("tags")
            )
            reply = (
                f"Updated note #{pending['note_id']} ('{updated['title']}')."
                if updated
                else f"Note #{pending['note_id']} no longer exists."
            )
        else:
            reply = "There was nothing pending to confirm."

        return {"messages": [AIMessage(content=reply)], "pending_confirmation": None}

    def agent_node(state: AgentState) -> Dict[str, Any]:
        prompt = SYSTEM_PROMPT
        hint = _INTENT_HINTS.get(state.get("last_intent"))
        if hint:
            prompt += f"\n\n[Hint: {hint}]"
        if state.get("pending_confirmation"):
            prompt += (
                f"\n\n[Note: there is a pending unconfirmed action -- "
                f"{state['pending_confirmation']['summary']}. If the user's message isn't a "
                "clear yes/no, ask them to clarify.]"
            )
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("execute_confirmation", execute_confirmation_node)

    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent", route_after_classify, {"execute_confirmation": "execute_confirmation", "agent": "agent"}
    )
    workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("execute_confirmation", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    return app, db
