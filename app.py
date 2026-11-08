import streamlit as st

st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Q&A Bot")

st.write(
    "Upload your documents and ask questions "
    "using Retrieval-Augmented Generation."
)

st.info("Project initialized successfully!")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")