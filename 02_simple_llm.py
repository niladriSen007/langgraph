from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import os

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_NEW")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY_NEW not found in environment variables.")

model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0
)


class LLMState(TypedDict):
    prompt: str
    response: str


def generate_reponse(state: LLMState) -> LLMState:
    question: str = state["prompt"]
    prompt = f"Answer the following question concisely:\n\n{question}"
    response = model.invoke(prompt).content
    state["response"] = response
    return state


graph = StateGraph(LLMState)
graph.add_node("generate_response", generate_reponse)

graph.add_edge(START, "generate_response")
graph.add_edge("generate_response", END)

workflow = graph.compile()

initial_state = {
    "prompt": "What is the capital of France?"
}
final_state = workflow.invoke(initial_state)

print(final_state["response"])
