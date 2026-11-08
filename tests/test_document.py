
from src.document_processing.pdf_loader import load_document

from langchain_core.documents import Document
from src.document_processing.text_splitter import split_documents



def test_txt_loader():
    text = b"This is a sample document for testing."

    documents = load_document(text, "sample.txt")

    assert len(documents) == 1
    assert documents[0].page_content == (
        "This is a sample document for testing."
    )
    assert documents[0].metadata["source"] == "sample.txt"
    assert documents[0].metadata["file_type"] == "txt"


def test_empty_txt():
    try:
        load_document(b"", "empty.txt")
        assert False, "Expected ValueError"
    except ValueError:
        pass



def test_document_chunking():
    text = "This is a sample sentence. " * 100

    documents = [
        Document(
            page_content=text,
            metadata={"source": "sample.txt", "page": 1}
        )
    ]

    chunks = split_documents(
        documents,
        chunk_size=200,
        chunk_overlap=40
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= 200
        assert chunk.metadata["source"] == "sample.txt"
        assert chunk.metadata["page"] == 1
        assert "chunk_id" in chunk.metadata
