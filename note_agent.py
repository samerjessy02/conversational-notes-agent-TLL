import datetime
import os
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Load environment variables (.env)
load_dotenv()


# =====================================================================
# 1. MOCK DATABASE LAYER
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
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.notes[self.next_id] = note
        self.next_id += 1
        return note

    def search_notes(self, query: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        results = list(self.notes.values())
        if tag:
            tag_clean = tag.lower().strip()
            results = [n for n in results if tag_clean in n["tags"]]
        if query:
            q_clean = query.lower()
            results = [
                n for n in results 
                if q_clean in n["title"].lower() or q_clean in n["body"].lower()
            ]
        return results

    def get_note(self, note_id: int) -> Optional[Dict[str, Any]]:
        return self.notes.get(note_id)

    def modify_note(self, note_id: int, title: Optional[str] = None, body: Optional[str] = None, tags: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
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


# Initialize Global DB and pre-populate mock data
db = NoteDatabase()
db.add_note("Team Standup", "Agreed to move standup to Tuesdays at 10 AM.", ["meetings", "team"])
db.add_note("API Redesign Notes", "Need to migrate endpoints from v1 REST to v2 GraphQL.", ["work", "api"])
db.add_note("Old Office Address", "123 Main St, Suite 400, New York, NY", ["personal", "address"])


# =====================================================================
# 2. TOOL SCHEMAS & TOOL IMPLEMENTATIONS
# =====================================================================
class AddNoteInput(BaseModel):
    title: str = Field(description="A concise, descriptive title for the note.")
    body: str = Field(description="The main text body of the note.")
    tags: Optional[List[str]] = Field(default=[], description="Keywords/categories for filtering.")

@tool("add_note", args_schema=AddNoteInput)
def add_note_tool(title: str, body: str, tags: Optional[List[str]] = None) -> str:
    """Create and store a new note in the system."""
    note = db.add_note(title, body, tags or [])
    return f"SUCCESS: Note #{note['id']} created. Title: '{note['title']}'."


class SearchNotesInput(BaseModel):
    query: Optional[str] = Field(None, description="Keyword or natural language text to search in note titles/body.")
    tag: Optional[str] = Field(None, description="Filter specifically by this tag name.")

@tool("search_notes", args_schema=SearchNotesInput)
def search_notes_tool(query: Optional[str] = None, tag: Optional[str] = None) -> str:
    """Retrieve existing notes by keyword, topic, tag, or natural language query."""
    results = db.search_notes(query=query, tag=tag)
    if not results:
        return "NO_MATCHES: No notes found matching your criteria."
    
    output = [f"Found {len(results)} matching note(s):"]
    for n in results:
        output.append(f"- ID #{n['id']}: [{n['title']}] Body: '{n['body']}' (Tags: {', '.join(n['tags'])})")
    return "\n".join(output)


class ModifyNoteInput(BaseModel):
    note_id: int = Field(description="The unique integer ID of the note to modify.")
    title: Optional[str] = Field(None, description="Updated title text.")
    body: Optional[str] = Field(None, description="Updated body content.")
    tags: Optional[List[str]] = Field(None, description="Updated tag list.")
    confirmed: bool = Field(False, description="Must be set to True ONLY after explicit user confirmation.")

@tool("modify_note", args_schema=ModifyNoteInput)
def modify_note_tool(note_id: int, title: Optional[str] = None, body: Optional[str] = None, tags: Optional[List[str]] = None, confirmed: bool = False) -> str:
    """Modify content, title, or tags of an existing note."""
    note = db.get_note(note_id)
    if not note:
        return f"ERROR: Note #{note_id} does not exist."

    if not confirmed:
        changes = []
        if title: changes.append(f"Title -> '{title}'")
        if body: changes.append(f"Body -> '{body}'")
        if tags: changes.append(f"Tags -> {tags}")
        change_desc = ", ".join(changes) if changes else "content updates"
        return f"REQUIRES_CONFIRMATION: Target Note #{note_id} ('{note['title']}'). Proposed changes: [{change_desc}]. Ask the user to confirm before proceeding."

    updated = db.modify_note(note_id, title=title, body=body, tags=tags)
    return f"SUCCESS: Note #{note_id} ('{updated['title']}') updated successfully."


class DeleteNoteInput(BaseModel):
    note_id: int = Field(description="The unique integer ID of the note to remove.")
    confirmed: bool = Field(False, description="Must be set to True ONLY after explicit user confirmation.")

@tool("delete_note", args_schema=DeleteNoteInput)
def delete_note_tool(note_id: int, confirmed: bool = False) -> str:
    """Delete a note permanently from storage."""
    note = db.get_note(note_id)
    if not note:
        return f"ERROR: Note #{note_id} does not exist."

    if not confirmed:
        return f"REQUIRES_CONFIRMATION: Action will permanently delete Note #{note_id} ('{note['title']}'). Ask the user for explicit confirmation."

    db.delete_note(note_id)
    return f"SUCCESS: Note #{note_id} ('{note['title']}') deleted permanently."


tools = [add_note_tool, search_notes_tool, modify_note_tool, delete_note_tool]


# =====================================================================
# 3. LANGGRAPH STATE & WORKFLOW DEFINITION
# =====================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    active_note_id: Optional[int]


SYSTEM_PROMPT = """You are an intelligent Conversational Note-Taking Agent. You manage personal notes entirely through natural language.

BEHAVIORAL RULES:
1. INTENT DISAMBIGUATION:
   - If a request matches multiple notes or is ambiguous (e.g. "update my meeting note"), execute `search_notes` first to inspect matching records.
   - If search returns multiple results, DO NOT guess or pick randomly. Show the user the options (IDs and Titles) and ask which one they mean.

2. CONFIRMATION ON DESTRUCTIVE ACTIONS:
   - Deleting or modifying a note requires explicit confirmation.
   - Call `modify_note` or `delete_note` with `confirmed=False` initially unless the user explicitly gave permission in their latest turn.
   - When the tool returns `REQUIRES_CONFIRMATION`, state what will change and ask the user for a clear 'Yes/No'.
   - ONLY pass `confirmed=True` after receiving user confirmation.

3. MULTI-TURN CONTEXT AWARENESS:
   - Handle context references like "that note", "add a tag to it", or "delete the last one" by utilizing recent conversation history and search tool output.
"""

# Switched from ChatOpenAI to ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> Dict[str, Any]:
    """Agent node that processes messages and invokes LLM reasoning."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Conditional router determining if tool execution is required."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# Compile with Checkpointer for Multi-Turn Session Memory
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# =====================================================================
# 4. CLI INTERFACE FOR TESTING
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Conversational Note-Taking Agent (Powered by Groq)")
    print(" Available test commands:")
    print("   - 'What did I write about the API?'")
    print("   - 'Update my standup note to say Wednesdays'")
    print("   - 'Delete the note about the old office address'")
    print("=" * 60 + "\n")

    config = {"configurable": {"thread_id": "session_1"}}

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input or user_input.lower() in ["exit", "quit"]:
                break

            input_state = {"messages": [HumanMessage(content=user_input)]}
            
            # Stream execution steps
            for event in app.stream(input_state, config=config, stream_mode="values"):
                last_msg = event["messages"][-1]
                
            if isinstance(last_msg, AIMessage) and last_msg.content:
                print(f"\nAgent: {last_msg.content}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")