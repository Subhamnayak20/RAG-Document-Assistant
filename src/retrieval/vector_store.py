
import chromadb
from langchain_core.documents import Document

from src.embeddings.embedding_model import generate_embeddings


class VectorStore:

    def __init__(self, persist_directory="chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: list[Document]):
        if not documents:
            raise ValueError("No documents provided.")

        texts = [doc.page_content for doc in documents]
        embeddings = generate_embeddings(texts)

        self.collection.add(
            ids=[
                f"chunk_{i}_{__import__('uuid').uuid4().hex}"
                for i in range(len(documents))
            ],
            documents=texts,
            embeddings=embeddings,
            metadatas=[doc.metadata for doc in documents]
        )

        return len(documents)

    def search(self, query: str, top_k: int = 4):
        from src.embeddings.embedding_model import (
            generate_query_embedding
        )

        query_embedding = generate_query_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )

        documents = []

        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            documents.append({
                "document": Document(
                    page_content=text,
                    metadata=metadata
                ),
                "distance": distance
            })

        return documents

    def clear(self):
        self.client.delete_collection("document_collection")
        self.collection = self.client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"}
        )
