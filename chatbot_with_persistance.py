from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_NEW")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY_NEW not found in environment variables.")

model = ChatOpenAI(
    api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0  # type: ignore
)


class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage], add_messages
    ]  ## add_messages is the reducer function


def chat_node(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot_node", chat_node)

# Add edges
graph_builder.add_edge(START, "chatbot_node")
graph_builder.add_edge("chatbot_node", END)

# Draw the graph
# graph.get_graph().draw_mermaid_png(output_file_path="chatbot_graph.png")
graph = graph_builder.compile()

while True:
    user_query = input("You: ")
    if user_query.strip().lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    response = graph.invoke({"messages": [HumanMessage(content=user_query)]})
    print(f"Response: {response['messages'][-1].content}")
