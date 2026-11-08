
from langchain_core.documents import Document
from src.retrieval.vector_store import VectorStore


def test_vector_store(tmp_path):
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma_test")
    )

    documents = [
        Document(
            page_content="Python is a programming language.",
            metadata={"source": "test.txt", "page": 1}
        ),
        Document(
            page_content="Machine learning uses data to learn patterns.",
            metadata={"source": "test.txt", "page": 2}
        )
    ]

    count = store.add_documents(documents)

    assert count == 2

    results = store.search("What is Python?", top_k=1)

    assert len(results) == 1
    assert "Python" in results[0]["document"].page_content
