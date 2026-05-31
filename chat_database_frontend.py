import streamlit as st
from langchain_core.messages import HumanMessage
from database_backend import chatbot, retrieve_all_threads, generate_thread_id
import uuid


# ---------------------------
# Session Init
# ---------------------------
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()


def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def reset_chat():
    new_id = generate_thread_id()
    st.session_state.thread_id = new_id
    st.session_state.message_history = []
    add_thread(new_id)


def load_conversation(thread_id):
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )

        if state and state.values:
            return state.values.get("messages", [])

    except Exception:
        pass

    return []


# Ensure current thread exists
add_thread(st.session_state.thread_id)


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()


st.sidebar.header("Conversations")

for tid in st.session_state.chat_threads[::-1]:
    if st.sidebar.button(
        f"Chat {tid[:8]}",
        key=f"thread_{tid}"
    ):
        st.session_state.thread_id = tid

        messages = load_conversation(tid)

        history = []

        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"

            history.append({
                "role": role,
                "content": msg.content
            })

        st.session_state.message_history = history


# ---------------------------
# Restore UI state if empty
# ---------------------------
if not st.session_state.message_history:
    messages = load_conversation(st.session_state.thread_id)

    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"

        st.session_state.message_history.append({
            "role": role,
            "content": msg.content
        })


# ---------------------------
# Render chat history
# ---------------------------
for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])


# ---------------------------
# Chat Input
# ---------------------------
user_input = st.chat_input("Type here")

if user_input:

    # show user message
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.text(user_input)


    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }


    def stream_response():
        for chunk, metadata in chatbot.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config,
            stream_mode="messages"
        ):
            if hasattr(chunk, "content"):
                yield chunk.content


    with st.chat_message("assistant"):
        ai_response = st.write_stream(stream_response())


    st.session_state.message_history.append({
        "role": "assistant",
        "content": ai_response
    })