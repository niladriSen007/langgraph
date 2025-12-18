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


class BlogState(TypedDict):
    topic: str
    outline: str
    draft: str


def create_blog_outline(state: BlogState) -> BlogState:
    topic: str = state["topic"]
    prompt = f"Create a detailed outline for a blog post about the following topic:\n\n{topic}"
    outline: str = model.invoke(prompt).content
    state["outline"] = outline
    return state


def create_blog(state: BlogState) -> BlogState:
    topic = state["topic"]
    outline = state["outline"]
    prompt = f"Write a comprehensive blog post based on the following outline about {topic}:\n\n{outline}"
    draft: str = model.invoke(prompt).content
    state["draft"] = draft
    return state
  
graph = StateGraph(BlogState)
graph.add_node("create_blog_outline", create_blog_outline)
graph.add_node("create_blog", create_blog)

graph.add_edge(START, "create_blog_outline")
graph.add_edge("create_blog_outline", "create_blog")
graph.add_edge("create_blog", END)

workflow = graph.compile()

initial_state = {
    "topic": "The Benefits of Learning Python Programming"
}
final_state = workflow.invoke(initial_state)
print("Blog Outline:\n", final_state["outline"])
print("\nBlog Draft:\n", final_state["draft"])