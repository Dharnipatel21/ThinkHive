from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid


class Chunker:
    """
    Splits large extracted text into smaller overlapping chunks
    suitable for embedding and vector retrieval.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk_text(self, text: str, document_id: str, page_number: int = 1) -> list[dict]:
        """
        Split text into overlapping chunks.

        Args:
            text: full extracted text from document
            document_id: MongoDB document ID
            page_number: page number text came from

        Returns:
            list of chunk dicts ready for embedding
        """
        if not text or not text.strip():
            return []

        raw_chunks = self.splitter.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            # Skip empty chunks
            if not chunk_text.strip():
                continue

            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "chunk_text": chunk_text.strip(),
                "document_id": document_id,
                "page_number": page_number,
                "chunk_index": i
            })

        return chunks

    def chunk_document_pages(self, pages: list[dict], document_id: str) -> list[dict]:
        """
        Chunk a multi-page document where each page is separate.

        Args:
            pages: list of dicts [{ page_number: int, text: str }]
            document_id: MongoDB document ID

        Returns:
            flat list of all chunks across all pages
        """
        all_chunks = []
        for page in pages:
            page_chunks = self.chunk_text(
                text=page["text"],
                document_id=document_id,
                page_number=page["page_number"]
            )
            all_chunks.extend(page_chunks)
        return all_chunks
