from dotenv import load_dotenv  # type: ignore
from langchain_core.messages import (  # type: ignore
    BaseMessage,
    HumanMessage,
)  # pyright: ignore[reportMissingImports]
from langchain_openai import ChatOpenAI  # type: ignore
from langgraph.graph.message import add_messages  # type: ignore
from typing_extensions import TypedDict, Annotated  # type: ignore
from langgraph.graph import StateGraph, START, END  # type: ignore
from langchain.tools import tool  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
from langchain.agents import create_agent  # type: ignore
from typing import Literal
from langgraph.prebuilt import ToolNode, tools_condition  # type: ignore
from langchain.messages import ToolMessage  # type: ignore
import os

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_NEW")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY_NEW not found in environment variables.")

model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)


class CalculatorInput(BaseModel):
    """Input for the calculator tool."""

    first_no: float = Field(..., description="The first number.")
    second_no: float = Field(..., description="The second number.")
    operation: Literal["add", "sub", "mul", "div"] = Field(
        ...,
        description="The operation to perform: addition, subtraction, multiplication, division.",
    )


@tool("calculator_tool", args_schema=CalculatorInput)
def calculator_tool(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """

    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


tools = [calculator_tool]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def llm_call(state: ChatState) -> str:
    """LLM decides to call a toolor not"""
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: ChatState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_args = tool_call["args"]
        result.append(ToolMessage(content=tool_args, tool_call_id=tool_call["id"]))
    return {"messages": result}


# def chat_node(state: StateGraph) -> str:
#     """LLM node that may answer or request a tool call."""
#     messages = state["messages"]
#     response = agent.invoke(messages)
#     return {"messages": [response]}


def check_chat_node_response_to_continue_or_end(
    state: ChatState,
) -> Literal["tool_node", END]:
    """Check if the last message from chat_node indicates a tool call is needed or to end the worlflow"""

    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    return END


graph = StateGraph(ChatState)
graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "llm_call")
graph.add_conditional_edges(
    "llm_call",
    check_chat_node_response_to_continue_or_end,
    ["tool_node", END],
)
graph.add_edge("tool_node", "llm_call")

chatbot = graph.compile()

result = chatbot.invoke(
    {
        "messages": [HumanMessage(content="What is the capital of France?")],
    }
)

for message in result["messages"]:
    message.pretty_print()
