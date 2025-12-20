from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, NotRequired, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
import os
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod  # type: ignore
from operator import add
from langgraph.graph.message import add_messages
from pprint import pprint
from langgraph.checkpoint.memory import InMemorySaver
import json

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_NEW")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY_NEW not found in environment variables.")

model = ChatOpenAI(
    api_key=OPENAI_API_KEY,  # type: ignore
    model="gpt-4o-mini",
    temperature=0
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    message = state["messages"]
    response = model.invoke(message)
    return {"messages": [response]}  # type: ignore


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Checkpointer
checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

initial_state = {
    "messages": [HumanMessage(content="What is the capital of France?")]
}


config = {"configurable": {"thread_id": "1"}}
result = chatbot.invoke(initial_state, config=config)
""" print(result["messages"])  # type: ignore """

print(chatbot.get_state_history(config=config))
