
import hashlib
import tempfile

import streamlit as st

from src.rag_pipeline import RAGPipeline


# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Assistant")
st.caption("Upload documents, ask questions, and get context-based answers.")


# ---------------------------
# Session initialization
# ---------------------------
def create_pipeline():
    directory = tempfile.mkdtemp(prefix="rag_assistant_")
    return RAGPipeline(persist_directory=directory)


if "pipeline" not in st.session_state:
    st.session_state.pipeline = create_pipeline()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------
# Sidebar: document upload
# ---------------------------
with st.sidebar:
    st.header("📁 Your Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button(
        "Process Documents",
        disabled=not uploaded_files,
        use_container_width=True
    ):
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            if file_hash in st.session_state.processed_files:
                st.info(f"{uploaded_file.name} is already processed.")
                continue

            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    result = st.session_state.pipeline.process_file(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name
                    )

                st.session_state.processed_files.add(file_hash)

                st.success(
                    f"{result['filename']}: "
                    f"{result['pages']} pages, "
                    f"{result['chunks']} chunks"
                )

            except Exception as error:
                st.error(f"Failed to process {uploaded_file.name}: {error}")

    st.divider()

    st.subheader("Session")

    st.write(
        f"Processed documents: "
        f"{len(st.session_state.processed_files)}"
    )

    if st.button("Clear Session", use_container_width=True):
        st.session_state.pipeline = create_pipeline()
        st.session_state.processed_files = set()
        st.session_state.messages = []
        st.rerun()


# ---------------------------
# Main chat interface
# ---------------------------
st.subheader("💬 Ask your documents")

if not st.session_state.processed_files:
    st.info("Upload and process a PDF or TXT document from the sidebar to get started.")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.write(
                        f"📄 {source['source']} — "
                        f"Page {source['page']}"
                    )


# Chat input
question = st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.processed_files
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating answer..."):
                result = st.session_state.pipeline.ask(question)

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            if sources:
                with st.expander("View sources"):
                    for source in sources:
                        st.write(
                            f"📄 {source['source']} — "
                            f"Page {source['page']}"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as error:
            error_message = f"Something went wrong: {error}"
            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })
