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


class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback."
    )


class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description="The category of issue mentioned in the feedback."
    )
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description='The emotional tone expressed by the user'
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description='How urgent or critical the issue appears to be'
    )


sentiment_model = model.with_structured_output(SentimentSchema)
diagnosis_model = model.with_structured_output(DiagnosisSchema)

prompt = 'What is the sentiment of the following review - The software too good'
sentiment_res = sentiment_model.invoke(prompt)
""" print(f"type(sentiment_res)", sentiment_res.sentiment) """


class FeedbackState(TypedDict):
    feedback: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


def find_sentiment(state: FeedbackState) -> dict[str, Literal["positive", "negative"]]:
    feedback: str = state["feedback"]
    prompt = f'What is the sentiment of the following review - {feedback}'
    sentiment_res = sentiment_model.invoke(prompt)
    state["sentiment"] = sentiment_res.sentiment
    return {"sentiment": state["sentiment"]}


def check_sentiment(state: FeedbackState) -> Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "negative":
        return "run_diagnosis"
    else:
        return "positive_response"


def _content_to_str(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def positive_response(state: FeedbackState) -> dict[str, str]:
    prompt = f"""Write a warm thank-you message in response to this review:
                \n\n\"{state['feedback']}\"\n
                Also, kindly ask the user to leave feedback on our website."""
    res = _content_to_str(model.invoke(prompt).content)
    state["response"] = res
    return {"response": state["response"]}


def run_diagnosis(state: FeedbackState):
    prompt = f"""Diagnose this negative review:\n\n{state['feedback']}\n"
    "Return issue_type, tone, and urgency.
    """
    diagnosis_res = diagnosis_model.invoke(prompt)
    state["diagnosis"] = diagnosis_res.model_dump()
    return {"diagnosis": state["diagnosis"]}


def negative_diagnosis(state: FeedbackState) -> dict[str, dict]:
    diagnosis_result = state["diagnosis"]
    prompt = f"""You are a support assistant.
    The user had a '{diagnosis_result['issue_type']}' issue, sounded '{diagnosis_result['tone']}', and marked urgency as '{diagnosis_result['urgency']}'.
    Write an empathetic, helpful resolution message.
    """
    res = _content_to_str(model.invoke(prompt).content)
    state["response"] = res
    return {"response": state["response"]}


graph = StateGraph(FeedbackState)
graph.add_node("find_sentiment", find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_diagnosis', negative_diagnosis)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_diagnosis")
graph.add_edge("negative_diagnosis", END)

workflow = graph.compile()

initial_state = {
    "feedback": "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}

# Pretty-print the final workflow state as a clean dictionary
result = workflow.invoke(initial_state)
try:
    # json gives the most compact, readable output (handles unicode nicely)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
except Exception:
    # Fallback to pprint if something isn't JSON serializable
    pprint(result, sort_dicts=False)


try:
    print("\nGenerating graph image...")
    graph_image = workflow.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER
    )
    with open("bmi_workflow_graph.png", "wb") as f:
        f.write(graph_image)
    print("✓ Graph saved as 'bmi_workflow_graph.png'")
except Exception as e:
    print(f"\nCouldn't generate PNG: {e}")
    # Alternative: save as ASCII art
    print("\nWorkflow graph (text representation):")
    print(workflow.get_graph().draw_ascii())
