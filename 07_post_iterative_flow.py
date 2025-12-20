from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, NotRequired, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import os
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod  # type: ignore
from operator import add
from pprint import pprint
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
