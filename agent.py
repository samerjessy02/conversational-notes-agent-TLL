"""
Core agent: database, tools, intent classifier, and the LangGraph workflow.

Both main.py (CLI) and ui.py (Streamlit) import build_app() from here so
there's exactly one implementation instead of two copies drifting apart.

DESIGN NOTES (read this before touching the graph)
----------------------------------------------------------------------------
1. Confirmation is enforced in CODE, not trusted to the LLM. modify_note_tool
   and delete_note_tool never mutate the database. They only look up the
   note and *stage* a proposed change into graph state. The only place a
   mutation actually happens is execute_confirmation_node, and the graph
   only reaches that node when classify_intent() has deterministically read
   a plain "yes"/"no" off the user's own next message. The LLM is not part
   of the confirmation-authority path.

2. Confirmations are queued, not single-slot. If the user (or the model,
   mid-conversation) stages a second destructive action before responding
   to the first, the first proposal is NOT silently discarded. Both live in
   `pending_confirmations`, a list acting as a stack: the most recently
   proposed action is the one a bare "yes"/"no" resolves (that's the
   question that was most recently "asked"), and after resolving it, if
   anything else is still queued, the reply says so and restates it.
   Proposing the same (tool, note_id) again replaces the earlier entry in
   place instead of queuing a duplicate.

3. Notes persist in SQLite (default: notes.db in the working directory,
   overridable via NOTES_DB_PATH). The seed data is only inserted the first
   time the file is created/empty -- on every later startup, whatever's on
   disk is used as-is, so notes survive a restart and the demo notes don't
   get duplicated on every run.
"""
import datetime
import os
import re
import sqlite3
import threading
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
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command

load_dotenv()


# =====================================================================
# 1. DATABASE LAYER (SQLite-backed, persists across restarts)
# =====================================================================
class NoteDatabase:
    """Persistent note storage backed by SQLite.

    Uses a single connection for the instance's lifetime (not one per
    thread) with a lock around every statement. This app is a single-user
    CLI/Streamlit prototype, not a high-concurrency service, so a lock is
    simpler and safer than per-thread connections -- the latter would
    silently give each thread its own isolated ":memory:" database, which
    is exactly the kind of bug that's invisible until it isn't.
    """

    def __init__(self, db_path: str = "notes.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["tags"] = [t for t in d["tags"].split(",") if t]
        return d

    def is_empty(self) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM notes").fetchone()
        return row["c"] == 0

    def add_note(self, title: str, body: str, tags: List[str]) -> Dict[str, Any]:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tags_str = ",".join(t.lower().strip() for t in tags)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO notes (title, body, tags, created_at) VALUES (?, ?, ?, ?)",
                (title, body, tags_str, now),
            )
            self._conn.commit()
            new_id = cur.lastrowid
        return self.get_note(new_id)

    def get_note(self, note_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def search_notes(self, query: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM notes ORDER BY id").fetchall()
        results = [self._row_to_dict(r) for r in rows]

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

    def modify_note(
        self, note_id: int, title: Optional[str] = None, body: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        existing = self.get_note(note_id)
        if not existing:
            return None
        new_title = title if title is not None else existing["title"]
        new_body = body if body is not None else existing["body"]
        new_tags = tags if tags is not None else existing["tags"]
        tags_str = ",".join(t.lower().strip() for t in new_tags)
        with self._lock:
            self._conn.execute(
                "UPDATE notes SET title=?, body=?, tags=? WHERE id=?", (new_title, new_body, tags_str, note_id)
            )
            self._conn.commit()
        return self.get_note(note_id)

    def delete_note(self, note_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            self._conn.commit()
        return cur.rowcount > 0


def make_seeded_db(db_path: Optional[str] = None) -> NoteDatabase:
    """Open (or create) the notes DB. Only inserts the demo notes the first
    time the file is empty/new -- an existing DB is used as-is, so restarts
    don't wipe or duplicate data.

    Pass db_path=":memory:" explicitly for tests -- an isolated, throwaway
    DB that never touches the real notes.db on disk. Tests must NOT rely on
    the default path, since (a) it would persist mutations (like a delete)
    across test runs, and (b) parallel test runs would collide on one file.
    """
    if db_path is None:
        db_path = os.environ.get("NOTES_DB_PATH", "notes.db")
    db = NoteDatabase(db_path)
    if db.is_empty():
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
    """Cheap, deterministic, regex-based intent tag for a user message."""
    t = text.strip()
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
    pending_confirmations: List[Dict[str, Any]]
    last_intent: Optional[str]


def _stage(queue: List[Dict[str, Any]], entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Push a proposal onto the confirmation queue. If a proposal already
    exists for the same (tool, note_id), replace it in place rather than
    stacking a duplicate -- e.g. the user asking to rename a note twice in a
    row before confirming should update the pending change, not queue two
    separate "rename note #3" confirmations."""
    queue = [q for q in queue if not (q["tool"] == entry["tool"] and q["note_id"] == entry["note_id"])]
    queue.append(entry)
    return queue


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
        state: Annotated[Dict, InjectedState] = None,
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
        entry = {
            "tool": "modify_note",
            "note_id": note_id,
            "title": title,
            "body": body,
            "tags": tags,
            "summary": f"update note #{note_id} ('{note['title']}'): {change_desc}",
        }
        queue = _stage(list((state or {}).get("pending_confirmations") or []), entry)
        return Command(
            update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)], "pending_confirmations": queue}
        )

    @tool("delete_note", args_schema=DeleteNoteInput)
    def delete_note_tool(
        note_id: int,
        state: Annotated[Dict, InjectedState] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = None,
    ) -> Any:
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
        entry = {
            "tool": "delete_note",
            "note_id": note_id,
            "summary": f"delete note #{note_id} ('{note['title']}')",
        }
        queue = _stage(list((state or {}).get("pending_confirmations") or []), entry)
        return Command(
            update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)], "pending_confirmations": queue}
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
acknowledge it. If the user stages more than one change before confirming either,
that's fine -- the system tracks all of them and resolves the most recent one first.
"""


# =====================================================================
# 6. GRAPH
# =====================================================================
def build_app(db: Optional[NoteDatabase] = None, checkpointer=None, llm=None):
    """Build the compiled graph. `db`, `checkpointer`, and `llm` are all
    injectable so the CLI, the UI, and tests can each supply their own
    (e.g. tests can pass a fake llm instead of hitting the real Groq API,
    and should always pass db=make_seeded_db(":memory:") for isolation).
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
        if state.get("last_intent") in ("confirm_yes", "confirm_no") and state.get("pending_confirmations"):
            return "execute_confirmation"
        return "agent"

    def execute_confirmation_node(state: AgentState) -> Dict[str, Any]:
        """The ONLY place a modify/delete is actually applied to the database.
        Reached only via a deterministic yes/no classification, never via the
        LLM deciding on its own that something is confirmed. Resolves the
        most recently staged item (top of the stack) first."""
        queue = list(state.get("pending_confirmations") or [])
        if not queue:
            return {"messages": [AIMessage(content="There's nothing pending to confirm.")]}

        pending = queue.pop()  # most recently proposed = resolved first

        if state["last_intent"] == "confirm_no":
            reply = f"Okay, cancelled -- I won't {pending['summary']}."
        elif pending["tool"] == "delete_note":
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

        if queue:
            next_item = queue[-1]
            reply += f" You still have a pending action: {next_item['summary']}. Reply yes/no to handle that one too."

        return {"messages": [AIMessage(content=reply)], "pending_confirmations": queue}

    def agent_node(state: AgentState) -> Dict[str, Any]:
        prompt = SYSTEM_PROMPT
        hint = _INTENT_HINTS.get(state.get("last_intent"))
        if hint:
            prompt += f"\n\n[Hint: {hint}]"
        queue = state.get("pending_confirmations") or []
        if queue:
            prompt += (
                f"\n\n[Note: there {'is' if len(queue) == 1 else 'are'} {len(queue)} pending unconfirmed "
                f"action(s), most recent: {queue[-1]['summary']}. If the user's message isn't a clear "
                "yes/no, ask them to clarify.]"
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
