from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
import streamlit as st
from backend import chatbot, retrive_all_threads
import uuid


####### Utility Function #######
def generate_thread_id():
    return str(uuid.uuid4())


def get_or_create_thread_id():
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = generate_thread_id()
    return st.session_state["thread_id"]


def reset_chat():
    st.session_state["message_history"] = []
    st.session_state["thread_id"] = generate_thread_id()
    add_thread_to_session(st.session_state["thread_id"])


def add_thread_to_session(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_specific_conversation(thread_id: str):
    return chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values[
        "messages"
    ]


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrive_all_threads()

add_thread_to_session(get_or_create_thread_id())

st.title("LangGraph Chatbot")
####### Sidebar UI ######3
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_specific_conversation(str(thread_id))

        response = []
        for message in messages:
            if isinstance(message, HumanMessage):
                response.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                response.append({"role": "assistant", "content": message.content})

        st.session_state["message_history"] = response


###### Main UI #######
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Say something")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # response_from_chatbot = chatbot.invoke(
    #     {"messages": [HumanMessage(content=user_input)]}, config=CONFIG
    # )
    # ai_message = response_from_chatbot["messages"][-1].content

    # message_history.append({"role": "assistant", "content": ai_message})
    # with st.chat_message("assistant"):
    #     st.markdown(ai_message)

    CONFIG = {"configurable": {"thread_id": get_or_create_thread_id()}}
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
