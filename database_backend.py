from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
import uuid


# ---------------------------
# LLM (Llama 3.1 via Ollama)
# ---------------------------
llm = ChatOllama(
    model="llama3.1",
    temperature=0.7
)


# ---------------------------
# State
# ---------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------
# Node
# ---------------------------
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ---------------------------
# DB + Checkpointer
# ---------------------------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)


# ---------------------------
# Graph
# ---------------------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ---------------------------
# Helpers
# ---------------------------
def retrieve_all_threads():
    threads = set()

    for checkpoint in checkpointer.list(None):
        thread_id = (
            checkpoint.config
            .get("configurable", {})
            .get("thread_id")
        )

        if thread_id:
            threads.add(thread_id)

    return sorted(list(threads))


def generate_thread_id():
    return str(uuid.uuid4())