"""
Demo UI — Streamlit chatbot for testing the RAG pipeline.
Run with: uv run streamlit run demo.py
"""
import streamlit as st
from indexing.doc_embedder import download_models
from generate import run_pipeline

st.title("NewsLLM 新聞助理")

# Load embedders once and cache them
@st.cache_resource
def load_embedders():
    return download_models()

dense_embedder, sparse_embedder = load_embedders()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
USER_ICON, AGENT_ICON = "./images/user.png", "./images/agents.png"
for msg in st.session_state.messages:
    avatar = USER_ICON if msg["role"] == "user" else AGENT_ICON
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("請輸入您的問題..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=USER_ICON):
        st.markdown(query)

    # Stream assistant response
    with st.chat_message("assistant", avatar=AGENT_ICON):
        response = st.write_stream(
            run_pipeline(query, dense_embedder, sparse_embedder)
        )

    st.session_state.messages.append({"role": "assistant", "content": response})