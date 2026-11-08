# 📚 RAG Document Assistant

**An AI-powered document question-answering application built using Retrieval-Augmented Generation (RAG), ChromaDB, Sentence Transformers, and NVIDIA Nemotron.**

Upload PDF or TXT documents and interact with their content through natural-language questions. The application retrieves relevant document passages using semantic search and generates context-aware answers using an LLM.

## 🚀 Live Demo

**Live Application:** Coming Soon

## ✨ Features

- **Multi-format document support:** Upload PDF and TXT files.
- **PDF text extraction:** Extract text and page-level metadata using PyMuPDF.
- **Intelligent text chunking:** Split documents into smaller, overlapping chunks for better retrieval.
- **Semantic search:** Generate embeddings using `all-MiniLM-L6-v2`.
- **Vector database:** Store and retrieve document embeddings using ChromaDB.
- **AI-powered answers:** Generate responses using NVIDIA Nemotron through its OpenAI-compatible API.
- **Source references:** Display the source filename and page number associated with retrieved passages.
- **Multi-turn Q&A:** Ask multiple questions about uploaded documents within the same session.
- **Session isolation:** Maintain separate document collections for individual application sessions.

## 🏗️ Architecture

The application follows a basic Retrieval-Augmented Generation pipeline.

```text
             User
              |
              v
       Streamlit Interface
              |
              v
       Upload PDF / TXT
              |
              v
       Document Extraction
           (PyMuPDF)
              |
              v
        Text Chunking
       (LangChain Splitter)
              |
              v
       Embedding Generation
    (Sentence Transformers)
              |
              v
          ChromaDB
       Vector Storage
              |
              |
       User Question
              |
              v
       Query Embedding
              |
              v
       Semantic Retrieval
          (ChromaDB)
              |
              v
       Retrieved Context
              |
              v
       NVIDIA Nemotron
              |
              v
       Contextual Answer
              |
              v
        Streamlit UI
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | NVIDIA Nemotron 3.5 Lightning |
| LLM API | NVIDIA Build API |
| Embedding Model | Sentence Transformers - all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF |
| Text Chunking | LangChain Text Splitters |
| API Client | OpenAI Python SDK |
| Testing | Pytest |

## 📂 Project Structure

```text
RAG-Document-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── src/
│   ├── __init__.py
│   │
│   ├── document_processing/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py
│   │   └── text_splitter.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_model.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── nemotron.py
│   │
│   └── rag_pipeline.py
│
└── tests/
    ├── __init__.py
    ├── test_document.py
    └── test_vectorstore.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Subhamnayak20/RAG-Document-Assistant.git

cd RAG-Document-Assistant
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key

NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
```

Get your API key from [NVIDIA Build](https://build.nvidia.com/).

**Important:** Never commit your `.env` file or expose your API key publicly.

### 5. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 🧪 Running Tests

Run the test suite using:

```bash
python -m pytest tests/ -v
```

Current test coverage includes:

- TXT document loading.
- Empty document validation.
- Document chunking.
- ChromaDB vector storage and semantic retrieval.

## ☁️ Deployment

The application is designed for deployment on **Streamlit Community Cloud**.

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Select `app.py` as the application entry point.
5. Add the NVIDIA API key and other environment variables under Streamlit Secrets.
6. Deploy the application.

Note: The current implementation uses session-specific temporary vector storage. Uploaded documents may need to be processed again when a session or application instance restarts.

## 🔮 Future Improvements

- Support for Word documents and multiple document formats.
- Improved conversational memory.
- Streaming LLM responses.
- Advanced retrieval techniques and reranking.
- Hybrid search combining semantic and keyword retrieval.
- Document summarization.
- Persistent cloud-based vector storage.
- Improved source citations and retrieval evaluation.

## 👨‍💻 Author

**Subham Nayak**

B.Tech – Computer Science and Engineering | 2026 Graduate

[GitHub](https://github.com/Subhamnayak20)

---

*Built to explore Retrieval-Augmented Generation, semantic search, vector databases, and LLM-powered applications.*