from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
import os
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod  # type: ignore

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_NEW")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY_NEW not found in environment variables.")

model = ChatOpenAI(
    api_key=OPENAI_API_KEY,  # type: ignore
    model="gpt-3.5-turbo",
    temperature=0
)


class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int

    strikeRate: float
    ballsPerBoundary: float
    boundaryPercent: float
    summary: str


def calculate_strike_rate(state: BatsmanState) -> dict[str, float]:
    runs: int = state["runs"]
    balls: int = state["balls"]
    strike_rate: float = (runs / balls) * 100 if balls > 0 else 0.0
    state["strikeRate"] = round(strike_rate, 2)
    return {"strikeRate": state["strikeRate"]}


def calculate_balls_per_boundary(state: BatsmanState) -> dict[str, float]:
    balls: int = state["balls"]
    fours: int = state["fours"]
    sixes: int = state["sixes"]
    total_boundaries: int = fours + sixes
    balls_per_boundary: float = (
        balls / total_boundaries) if total_boundaries > 0 else float('inf')
    state["ballsPerBoundary"] = round(balls_per_boundary, 2)
    return {"ballsPerBoundary": state["ballsPerBoundary"]}


def calculate_boundary_percent(state: BatsmanState) -> dict[str, float]:
    runs: int = state["runs"]
    fours: int = state["fours"]
    sixes: int = state["sixes"]
    boundary_runs: int = (fours * 4) + (sixes * 6)
    boundary_percent: float = (boundary_runs / runs) * 100 if runs > 0 else 0.0
    state["boundaryPercent"] = round(boundary_percent, 2)
    return {"boundaryPercent": state["boundaryPercent"]}


def generate_summary(state: BatsmanState) -> dict[str, str]:
    runs: int = state["runs"]
    balls: int = state["balls"]
    fours: int = state["fours"]
    sixes: int = state["sixes"]

    prompt = (
        f"Generate a brief summary for a batsman who scored {state['runs']} runs off {state['balls']} balls, "
        f"including {state['fours']} fours and {state['sixes']} sixes. The batsman's strike rate is {state['strikeRate']}, "
        f"balls per boundary is {state['ballsPerBoundary']}, and boundary percentage is {state['boundaryPercent']}."
    )

    summary: str = model.invoke(prompt).content  # type: ignore
    state["summary"] = summary
    return {"summary": state["summary"]}


graph = StateGraph(BatsmanState)

graph.add_node("calculate_strike_rate", calculate_strike_rate)
graph.add_node("calculate_balls_per_boundary", calculate_balls_per_boundary)
graph.add_node("calculate_boundary_percent", calculate_boundary_percent)
graph.add_node("generate_summary", generate_summary)

graph.add_edge(START, "calculate_strike_rate")
graph.add_edge(START, "calculate_balls_per_boundary")
graph.add_edge(START, "calculate_boundary_percent")

graph.add_edge("calculate_strike_rate", "generate_summary")
graph.add_edge("calculate_balls_per_boundary", "generate_summary")
graph.add_edge("calculate_boundary_percent", "generate_summary")

graph.add_edge("generate_summary", END)

workflow = graph.compile()

initial_state = {
    "runs": 85,
    "balls": 60,
    "fours": 8,
    "sixes": 3
}

final_state = workflow.invoke(initial_state)  # type: ignore
print("Final State:\n", final_state)

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


# in parallel workflow we have multiple nodes running in parallel so we won't be able to return the state
# from individual functions. Hence we return only the calculated values as dict and
# the StateGraph will take care of merging them into the main state.
