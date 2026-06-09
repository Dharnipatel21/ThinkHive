from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    Converts text into vector embeddings using multilingual-e5-large.
    Used during ingestion (chunks) and query time (user queries).
    """

    def __init__(self):
        print("Loading multilingual-e5-large model... (first run downloads ~2.2GB)")
        self.model = SentenceTransformer("intfloat/multilingual-e5-large")
        self.embedding_dim = 1024
        print("Model loaded successfully.")

    def generate_embeddings(self, chunks: list[dict]) -> list[dict]:
        """
        Embed a list of chunk dicts. Adds 'embedding' key to each.

        Args:
            chunks: list of dicts with at least 'chunk_text' key

        Returns:
            Same list with 'embedding' key added to each dict
        """
        if not chunks:
            return chunks

        # multilingual-e5-large requires "passage: " prefix during indexing
        texts = [f"passage: {chunk['chunk_text']}" for chunk in chunks]

        # Batch encode for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()

        return chunks

    def embed_query(self, query_text: str) -> list[float]:
        """
        Embed a single user query string.

        Args:
            query_text: plain text user question

        Returns:
            list of floats (length 1024)
        """
        # multilingual-e5-large requires "query: " prefix during retrieval
        prefixed = f"query: {query_text}"
        embedding = self.model.encode(prefixed, convert_to_numpy=True)
        return embedding.tolist()
