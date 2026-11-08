
from src.document_processing.pdf_loader import load_document
from src.document_processing.text_splitter import split_documents
from src.retrieval.vector_store import VectorStore
from src.llm.nemotron import generate_answer


class RAGPipeline:
    def __init__(self, persist_directory="chroma_db"):
        self.vector_store = VectorStore(
            persist_directory=persist_directory
        )

    def process_file(self, file_bytes: bytes, filename: str):
        # 1. Extract text
        documents = load_document(file_bytes, filename)

        # 2. Split into chunks
        chunks = split_documents(documents)

        # 3. Generate embeddings and store in ChromaDB
        stored_count = self.vector_store.add_documents(chunks)

        return {
            "filename": filename,
            "pages": len(documents),
            "chunks": stored_count
        }

    def ask(self, question: str, top_k: int = 4):
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if self.vector_store.collection.count() == 0:
            raise ValueError("Please upload and process a document first.")

        # 1. Retrieve relevant chunks
        results = self.vector_store.search(
            query=question,
            top_k=top_k
        )

        if not results:
            return {
                "answer": "I couldn't find relevant information in the document.",
                "sources": []
            }

        # 2. Build context for the LLM
        context_parts = []
        sources = []

        for result in results:
            document = result["document"]
            metadata = document.metadata

            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")

            context_parts.append(
                f"Source: {source}, Page: {page}\n"
                f"{document.page_content}"
            )

            sources.append({
                "source": source,
                "page": page,
                "distance": result["distance"]
            })

        context = "\n\n".join(context_parts)

        # 3. Generate answer using Nemotron
        answer = generate_answer(
            question=question,
            context=context
        )

        return {
            "answer": answer,
            "sources": sources
        }
