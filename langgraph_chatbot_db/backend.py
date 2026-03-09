from typing import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
import os
import sqlite3

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_NEW")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY_NEW not found in environment variables.")

model = ChatOpenAI(
    api_key=OPENAI_API_KEY, model="gpt-5.1", temperature=0  # type: ignore
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


sqlite_connection = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=sqlite_connection)

graph = StateGraph(ChatState)
graph.add_node("chatbot_node", chat_node)
graph.add_edge(START, "chatbot_node")
graph.add_edge("chatbot_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_threads():
    thread_list = set()
    for checkpoint in checkpointer.list(None):
        thread_list.add(checkpoint.config["configurable"]["thread_id"])
    return list(thread_list)
