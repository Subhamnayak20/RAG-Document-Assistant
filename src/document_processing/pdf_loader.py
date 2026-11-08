
import pymupdf 
from langchain_core.documents import Document


def load_pdf(file_bytes: bytes, filename: str) -> list[Document]:
    """
    Extract text from a PDF.

    Each PDF page becomes a separate Document,
    preserving its page number in metadata.
    """

    documents = []

    try:
        pdf = pymupdf.open(stream=file_bytes, filetype="pdf")

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                            "file_type": "pdf"
                        }
                    )
                )

        pdf.close()

    except Exception as error:
        raise ValueError(f"Could not read PDF: {error}") from error

    if not documents:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be empty or contain scanned images."
        )

    return documents


def load_txt(file_bytes: bytes, filename: str) -> list[Document]:
    """Extract text from a TXT file."""

    try:
        text = file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("TXT file must use UTF-8 encoding.") from error

    if not text:
        raise ValueError("The TXT file is empty.")

    return [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "page": 1,
                "file_type": "txt"
            }
        )
    ]


def load_document(file_bytes: bytes, filename: str) -> list[Document]:
    """Route the uploaded file to the appropriate loader."""

    extension = filename.lower().rsplit(".", 1)[-1]

    if extension == "pdf":
        return load_pdf(file_bytes, filename)

    if extension == "txt":
        return load_txt(file_bytes, filename)

    raise ValueError(f"Unsupported file type: .{extension}")
