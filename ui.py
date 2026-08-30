import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# CSS Fix: Keep header visible so sidebar toggle works, but remove header background clutter
st.markdown("""
<style>
    /* Keeps the sidebar toggle button visible while hiding top padding/background */
    [data-testid="stHeader"] {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Conversational RAG Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = "user-session-1"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar (with toggle restored)
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.session_id = st.text_input("Session ID", value=st.session_state.session_id)
    if st.button("Clear Chat Window"):
        st.session_state.messages = []
        st.rerun()

# Render chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Bottom Input Area with Inline Attachment
st.divider()

col1, col2 = st.columns([1, 10])

with col1:
    # Popover acts like an "Attach File" button right by the prompt!
    with st.popover("📎 Upload"):
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], key="inline_pdf")
        if uploaded_file is not None and st.button("Index PDF"):
            with st.spinner("Indexing PDF..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
                    if res.status_code == 200:
                        st.success(res.json().get("message"))
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

with col2:
    prompt = st.chat_input("Ask a question about your documents...")

# Handle Prompt Logic
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": prompt},
                    timeout=120
                )
                if res.status_code == 200:
                    answer = res.json()
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")