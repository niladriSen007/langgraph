from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod


class BMIState(TypedDict):
    weight: float  # in kilograms
    height: float  # in meters
    bmi: float     # Body Mass Index


def calculate_bmi(state: BMIState) -> BMIState:
    weight: float = state["weight"]
    height: float = state["height"]
    bmi: float = weight / (height ** 2)

    state["bmi"] = round(bmi, 2)
    return state


def label_bmmi_report(state: BMIState) -> BMIState:
    bmi: float = state["bmi"]
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        category = "Normal weight"
    elif 25 <= bmi < 29.9:
        category = "Overweight"
    else:
        category = "Obesity"

    state["bmi_category"] = category
    return state


graph = StateGraph(BMIState)
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("label_bmmi_report", label_bmmi_report)

graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "label_bmmi_report")
graph.add_edge("label_bmmi_report", END)

workflow = graph.compile()

# execute the graph
initial_state = {
    "weight": 70.0,  # example weight in kg
    "height": 1.75   # example height in meters
}

final_state = workflow.invoke(initial_state)
print(final_state)

# Save the graph as PNG file using local rendering
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
