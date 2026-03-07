from langchain_core.messages import HumanMessage
import streamlit as st
from backend import chatbot

st.title("LangGraph Chatbot")

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

message_history: list[dict[str, str]] = st.session_state["message_history"]

for message in message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Say something")

if user_input:
    message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # response_from_chatbot = chatbot.invoke(
    #     {"messages": [HumanMessage(content=user_input)]}, config=CONFIG
    # )
    # ai_message = response_from_chatbot["messages"][-1].content

    # message_history.append({"role": "assistant", "content": ai_message})
    # with st.chat_message("assistant"):
    #     st.markdown(ai_message)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )
    message_history.append({"role": "assistant", "content": ai_message})
