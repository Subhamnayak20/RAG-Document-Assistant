
from src.retrieval.vector_store import VectorStore


def retrieve_documents(
    query: str,
    vector_store: VectorStore,
    top_k: int = 4
):
    return vector_store.search(query, top_k)
