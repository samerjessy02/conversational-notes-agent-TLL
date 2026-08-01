"""
Core agent: database, tools, intent classifier, and the LangGraph workflow.

Both main.py (CLI) and ui.py (Streamlit) import build_app() from here so
there's exactly one implementation instead of two copies drifting apart.

DESIGN NOTES (read this before touching the graph)
----------------------------------------------------------------------------
1. Confirmation is a real human-in-the-loop pause, not text parsing. When
   the agent proposes a modify_note/delete_note call, the graph routes to
   human_review_node, which calls LangGraph's interrupt(). This freezes
   execution -- app.invoke() returns immediately with a `__interrupt__` key
   and a structured payload describing exactly what would change. Nothing
   runs until the caller resumes with Command(resume={"approved": bool}).
   The CLI prompts for y/n; the Streamlit UI renders actual Yes/No buttons.
   Either way, the decision is a boolean the caller supplies explicitly --
   there's no "the model decided this looked like a yes" step anymore.

2. modify_note_tool and delete_note_tool mutate the database directly, same
   as add_note_tool. That's safe now, specifically because by the time
   ToolNode ever gets to run them, human_review_node has already gated the
   call on interrupt()+approval. The tools themselves don't need to know
   anything about confirmation -- the gate lives entirely in the graph's
   routing, one level up.

3. Because interrupt() genuinely blocks the graph, there's no need for a
   confirmation queue anymore: the user (or the model) literally cannot
   start a second turn while one destructive action is awaiting approval --
   the graph is paused. The one edge case handled explicitly is a single
   agent turn proposing MORE THAN ONE destructive tool call at once (e.g.
   the model decides to delete note A and rename note B in the same
   message): those are bundled into a single interrupt payload with a list
   of changes, and approval/rejection applies to all of them together. This
   keeps tool_call/tool_result ordering simple and avoids having to
   reconstruct a partially-approved AIMessage (which is exactly the kind of
   thing that trips up strict OpenAI-style tool-calling APIs like Groq's).

4. Notes persist in SQLite (default: notes.db in the working directory,
   overridable via NOTES_DB_PATH). Seed data is only inserted the first
   time the file is created/empty.
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
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

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
    silently give each thread its own isolated ":memory:" database.
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
    DB that never touches the real notes.db on disk.
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
# 2. INTENT CLASSIFIER (hint-only now -- confirmation no longer depends on it)
# =====================================================================
_DELETE_RE = re.compile(r"\b(delete|remove|erase|trash|get rid of)\b", re.I)
_MODIFY_RE = re.compile(r"\b(update|change|edit|modify|rename|correct|fix)\b", re.I)
_CREATE_RE = re.compile(r"\b(add|create|new note|jot down|write down|save a note|note that|remember that)\b", re.I)
_SEARCH_RE = re.compile(r"\b(find|search|show|list|display|what did i|look ?up|recall|any notes)\b", re.I)

Intent = Literal["delete", "modify", "create", "search", "chitchat"]


def classify_intent(text: str) -> Intent:
    """Cheap, deterministic, regex-based intent tag for a user message. Used
    only as a soft hint injected into the system prompt -- the model can
    ignore it. It is NOT involved in the confirmation flow anymore (see
    human_review_node / interrupt() below for that)."""
    t = text.strip()
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
    last_intent: Optional[str]
    rejected_this_turn: List[str]
    # "tool:note_id" keys the user has already said no to since their last
    # actual chat message. Scoped to "this turn" because it's reset in
    # classify_intent_node, which only runs when a NEW HumanMessage arrives
    # -- a resumed interrupt never passes back through it. This is what lets
    # the user retry later by just asking again, while preventing the model
    # from re-proposing the identical rejected change in a loop right now.


# =====================================================================
# 4. TOOLS
# =====================================================================
DESTRUCTIVE_TOOLS = {"modify_note", "delete_note"}


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
        note_id: int, title: Optional[str] = None, body: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> str:
        """Modify an existing note's title, body, or tags. Only ever reached
        AFTER the user has explicitly approved via the confirmation prompt --
        this tool just applies the change."""
        updated = db.modify_note(note_id, title=title, body=body, tags=tags)
        if not updated:
            return f"ERROR: Note #{note_id} does not exist."
        return f"SUCCESS: Note #{note_id} ('{updated['title']}') updated."

    @tool("delete_note", args_schema=DeleteNoteInput)
    def delete_note_tool(note_id: int) -> str:
        """Delete a note permanently. Only ever reached AFTER the user has
        explicitly approved via the confirmation prompt."""
        note = db.get_note(note_id)
        if not note:
            return f"ERROR: Note #{note_id} does not exist."
        db.delete_note(note_id)
        return f"SUCCESS: Note #{note_id} ('{note['title']}') deleted permanently."

    return [add_note_tool, search_notes_tool, modify_note_tool, delete_note_tool]


def _describe_change(tool_name: str, args: Dict[str, Any], note: Optional[Dict[str, Any]]) -> str:
    if not note:
        return f"Note #{args.get('note_id')} was not found (it may already be deleted)."
    if tool_name == "delete_note":
        return f"Delete note #{note['id']} (\"{note['title']}\") permanently? This cannot be undone."
    changes = []
    for field in ("title", "body", "tags"):
        if args.get(field) is not None:
            changes.append(f"{field}: {note[field]!r} -> {args[field]!r}")
    change_str = "; ".join(changes) if changes else "(no field changes specified)"
    return f"Update note #{note['id']} (\"{note['title']}\")? {change_str}"


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
need to ask for confirmation yourself. The system automatically pauses after your
tool call and shows the user a Yes/No prompt; it only applies the change if they
approve it. If you see a ToolMessage saying an action was cancelled, just
acknowledge it and ask what they'd like to do instead.

ACT, DON'T NARRATE
Never say you "will now" do something, or that you're "going ahead" with an
action, unless you are calling the tool for it in this exact same response.
Only an actual tool call triggers the confirmation the user needs to see --
a sentence promising future action does nothing on its own and will confuse
the user about whether anything is actually pending. If you intend to act,
call the tool now. If you're not ready to act, ask a direct question instead.

BATCH MULTI-NOTE REQUESTS
If the user asks to change or delete more than one note in a single message,
resolve every note ID you need (searching first if necessary), then call
modify_note / delete_note for ALL of them in this same response. This lets
the system show one combined confirmation instead of asking about each note
separately across multiple turns.
"""


# =====================================================================
# 6. GRAPH
# =====================================================================
def build_app(db: Optional[NoteDatabase] = None, checkpointer=None, llm=None):
    """Build the compiled graph. `db`, `checkpointer`, and `llm` are all
    injectable so the CLI, the UI, and tests can each supply their own
    (e.g. tests can pass a fake llm instead of hitting the real Groq API,
    and should always pass db=NoteDatabase(":memory:") for isolation).
    Returns (app, db).

    A checkpointer is required for interrupt()/Command(resume=...) to work
    at all -- it's how LangGraph knows where execution paused for a given
    thread_id. Defaults to MemorySaver if not supplied.
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
        return {"last_intent": intent, "rejected_this_turn": []}

    def agent_node(state: AgentState) -> Dict[str, Any]:
        prompt = SYSTEM_PROMPT
        hint = _INTENT_HINTS.get(state.get("last_intent"))
        if hint:
            prompt += f"\n\n[Hint: {hint}]"
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> Literal["human_review", "tools", "__end__"]:
        last = state["messages"][-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return END
        if any(tc["name"] in DESTRUCTIVE_TOOLS for tc in last.tool_calls):
            return "human_review"
        return "tools"

    def human_review_node(state: AgentState) -> Dict[str, Any]:
        """The ONLY place a modify/delete gets a chance to run. Pauses the
        entire graph via interrupt() and waits for the caller (CLI prompt or
        UI Yes/No buttons) to resume with an explicit approved: bool. If more
        than one destructive tool call is in this turn, they're bundled into
        one payload and approved/rejected together.

        Circuit breaker: if the model re-proposes the EXACT same (tool,
        note_id) that was already rejected earlier in this turn, we do not
        interrupt again -- we auto-cancel it and end the turn with a fixed
        message, without going back through the LLM. This exists because
        nothing guarantees a model accepts a rejection gracefully; a model
        that keeps re-proposing after "no" would otherwise force the human
        to keep clicking "No" indefinitely (observed in practice). Ending
        the turn outright, rather than looping back to "agent", is what
        actually guarantees termination -- looping back just gives the model
        another chance to propose it again.
        """
        last = state["messages"][-1]
        destructive_calls = [tc for tc in last.tool_calls if tc["name"] in DESTRUCTIVE_TOOLS]
        already_rejected = set(state.get("rejected_this_turn") or [])
        keys = [f"{tc['name']}:{tc['args'].get('note_id')}" for tc in destructive_calls]

        if any(k in already_rejected for k in keys):
            cancel_messages = [
                ToolMessage(
                    content="CANCELLED: this exact change was already rejected earlier in this "
                    "conversation turn and will not be asked about again automatically.",
                    tool_call_id=tc["id"],
                )
                for tc in last.tool_calls
            ]
            final_msg = AIMessage(
                content="That action was already declined, so I won't ask again. Let me know if "
                "there's something else you'd like to do."
            )
            return {"messages": cancel_messages + [final_msg]}

        changes = []
        for tc in destructive_calls:
            note = db.get_note(tc["args"].get("note_id"))
            changes.append(
                {
                    "tool": tc["name"],
                    "note_id": tc["args"].get("note_id"),
                    "summary": _describe_change(tc["name"], tc["args"], note),
                }
            )

        decision = interrupt({"type": "confirmation_required", "changes": changes})
        approved = bool(isinstance(decision, dict) and decision.get("approved"))

        if approved:
            return {}  # leave the AIMessage's tool_calls untouched; "tools" will run them for real

        cancel_messages = [
            ToolMessage(content="CANCELLED: user did not approve this action.", tool_call_id=tc["id"])
            for tc in last.tool_calls
        ]
        return {"messages": cancel_messages, "rejected_this_turn": list(already_rejected) + keys}

    def route_after_human_review(state: AgentState) -> Literal["tools", "agent", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            # The circuit breaker above ends with an AIMessage that has no
            # tool_calls -- that's the "stop here" signal.
            return "tools" if last.tool_calls else END
        # A plain rejection (first time) ends on a ToolMessage -- let the
        # agent see it and decide what to do next (e.g. try a different note).
        return "agent"

    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("human_review", human_review_node)

    workflow.add_edge(START, "classify_intent")
    workflow.add_edge("classify_intent", "agent")
    workflow.add_conditional_edges(
        "agent", route_after_agent, {"human_review": "human_review", "tools": "tools", END: END}
    )
    workflow.add_conditional_edges(
        "human_review", route_after_human_review, {"tools": "tools", "agent": "agent", END: END}
    )
    workflow.add_edge("tools", "agent")

    if checkpointer is None:
        checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    return app, db
