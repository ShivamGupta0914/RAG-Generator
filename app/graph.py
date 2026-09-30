import uuid
from threading import RLock
from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from app.config import RETRIEVAL_K
from app.llm import get_chat_model
from app.store import list_files, search_documents

SYSTEM_MESSAGE = (
    "Answer from the uploaded documents. Use the rag tool. "
    "If the documents do not have the answer, say you don't know. "
    "Do not add information that is not in the documents."
)


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


thread_id = "chat"
active_chat_id: str | None = None
chats: dict[str, dict] = {}
_clock = 0
_lock = RLock()


def _touch(chat: dict):
    global _clock
    _clock += 1
    chat["touched"] = _clock


def _public(chat: dict) -> dict:
    return {
        "id": chat["id"],
        "title": chat["title"],
        "thread_id": chat["thread_id"],
    }


def ensure_chat() -> dict:
    global active_chat_id, thread_id
    if active_chat_id and active_chat_id in chats:
        thread_id = chats[active_chat_id]["thread_id"]
        return chats[active_chat_id]
    return create_chat()


def create_chat() -> dict:
    global active_chat_id, thread_id
    chat_id = uuid.uuid4().hex[:8]
    thread_id = uuid.uuid4().hex[:8]
    chat = {"id": chat_id, "title": "New chat", "thread_id": thread_id}
    _touch(chat)
    chats[chat_id] = chat
    active_chat_id = chat_id
    return chat


def select_chat(chat_id: str) -> dict:
    global active_chat_id, thread_id
    chat = chats.get(chat_id)
    if chat is None:
        raise KeyError("That chat is not open.")
    active_chat_id = chat_id
    thread_id = chat["thread_id"]
    _touch(chat)
    return chat


def rename_chat(chat_id: str, title: str) -> dict:
    chat = chats.get(chat_id)
    if chat is None:
        raise KeyError("That chat is not open.")
    cleaned = " ".join(title.split()).strip()
    if not cleaned:
        raise ValueError("Give the chat a title.")
    chat["title"] = cleaned[:80]
    return chat


def delete_chat(chat_id: str) -> dict:
    global active_chat_id, thread_id
    if chat_id not in chats:
        raise KeyError("That chat is not open.")
    del chats[chat_id]
    if active_chat_id == chat_id:
        active_chat_id = None
        if chats:
            newest = max(chats.values(), key=lambda item: item["touched"])
            active_chat_id = newest["id"]
            thread_id = newest["thread_id"]
        else:
            create_chat()
    return chats[active_chat_id]


def reset_chat():
    """A new upload starts a new thread. Files already in Chroma stay there."""
    global thread_id
    chat = ensure_chat()
    thread_id = uuid.uuid4().hex[:8]
    chat["thread_id"] = thread_id
    chat["title"] = "New chat"
    _touch(chat)


def list_chats() -> list[dict]:
    ordered = sorted(chats.values(), key=lambda item: item["touched"], reverse=True)
    return [_public(chat) for chat in ordered]


@tool
def rag_tool(query: str) -> str:
    """Search the documents the user uploaded. Return the filename and the matching passage."""
    query = (query or "").strip()
    if not query:
        return "No matching text in the uploaded documents."

    docs = search_documents(query, k=RETRIEVAL_K)
    if not docs:
        return "No matching text in the uploaded documents."

    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "file")
        parts.append(f"{source}\n{doc.page_content}")
    return "\n\n".join(parts)


def chat_node(state: MessageState):
    messages = [SystemMessage(content=SYSTEM_MESSAGE), *state["messages"]]
    response = get_chat_model().bind_tools([rag_tool]).invoke(messages)
    return {"messages": [response]}


def build_graph():
    graph = StateGraph(MessageState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode([rag_tool]))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    return graph.compile(checkpointer=MemorySaver())


workflow = build_graph()


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or "")
        return "".join(parts)
    return str(content or "")


def visible_messages(messages) -> list[dict]:
    thread = []
    for message in messages:
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            thread.append({"role": "user", "content": _text(message.content)})
        elif getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", None):
            text = _text(message.content).strip()
            if text:
                thread.append({"role": "assistant", "content": text})
    return thread


def read_thread() -> list[dict]:
    state = workflow.get_state({"configurable": {"thread_id": thread_id}})
    messages = []
    if state and state.values:
        messages = state.values.get("messages") or []
    return visible_messages(messages)


def ask(question: str, chat_id: str | None = None) -> list[dict]:
    global thread_id
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("Enter a question.")

    with _lock:
        if chat_id:
            select_chat(chat_id)
        else:
            ensure_chat()

        chat = chats[active_chat_id]
        if chat["title"] == "New chat":
            chat["title"] = cleaned[:48] + ("…" if len(cleaned) > 48 else "")
        _touch(chat)

        result = workflow.invoke(
            {"messages": [HumanMessage(content=cleaned)]},
            {"configurable": {"thread_id": thread_id}},
        )
        return visible_messages(result["messages"])


def snapshot() -> dict:
    with _lock:
        ensure_chat()
        return {
            "active_chat_id": active_chat_id,
            "chats": list_chats(),
            "files": list_files(),
            "messages": read_thread(),
        }
