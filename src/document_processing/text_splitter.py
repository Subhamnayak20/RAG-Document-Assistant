
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> list[Document]:
    """
    Split documents into smaller overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    if not documents:
        raise ValueError("No documents provided for splitting.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Add a unique chunk index to each chunk's metadata.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks
