from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
from ..processing.embedder import Embedder

ROLE_ACCESS = {
    "super_admin": ["public", "internal", "restricted", "confidential"],
    "org_admin":   ["public", "internal", "restricted", "confidential"],
    "manager":     ["public", "internal", "restricted"],
    "employee":    ["public", "internal"],
    "guest":       ["public"]
}


class Retriever:
    """
    Retrieves semantically relevant chunks from Qdrant.
    Applies org isolation, role-based classification filter,
    and document weight to rank results.
    """

    def __init__(self, qdrant_client: QdrantClient, embedder: Embedder, collection_name: str = "knowledgeai_chunks"):
        self.qdrant = qdrant_client
        self.embedder = embedder
        self.collection_name = collection_name

    def retrieve(self, query: str, org_id: str, user_role: str, top_k: int = 5, document_ids: list = None) -> list[dict]:
        """
        Retrieve top-K relevant chunks for a query.

        Args:
            query: user's question in English
            org_id: organisation ID for data isolation
            user_role: user role for classification filtering
            top_k: number of chunks to retrieve
            document_ids: optional list to restrict search to specific docs

        Returns:
            list of chunk dicts sorted by weighted score
        """
        # Embed the query
        query_vector = self.embedder.embed_query(query)

        # Build classification filter from role
        allowed_classifications = ROLE_ACCESS.get(user_role, ["public"])

        # Build Qdrant filter
        must_conditions = [
            FieldCondition(key="org_id", match=MatchValue(value=org_id)),
            FieldCondition(
                key="classification",
                match=MatchAny(any=allowed_classifications)
            )
        ]

        # Optional document ID filter
        if document_ids:
            must_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=document_ids)
                )
            )

        qdrant_filter = Filter(must=must_conditions)

        # Search Qdrant
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )

        chunks = []
        for result in results:
            payload = result.payload
            similarity_score = result.score
            retrieval_weight = payload.get("retrieval_weight", 1.0)

            # Apply document weight to similarity score
            weighted_score = round(similarity_score * retrieval_weight, 4)

            chunks.append({
                "chunk_id": str(result.id),
                "chunk_text": payload.get("chunk_text", ""),
                "document_id": payload.get("document_id", ""),
                "document_name": payload.get("document_name", ""),
                "page_number": payload.get("page_number", 1),
                "chunk_index": payload.get("chunk_index", 0),
                "classification": payload.get("classification", "internal"),
                "similarity_score": round(similarity_score, 4),
                "retrieval_weight": retrieval_weight,
                "weighted_score": weighted_score,
                "age_tag": payload.get("age_tag", "unknown"),
                "freshness_tag": payload.get("freshness_tag", "unverified"),
                "usage_tag": payload.get("usage_tag", "unused")
            })

        # Sort by weighted score
        chunks.sort(key=lambda x: x["weighted_score"], reverse=True)
        return chunks

    def delete_document(self, document_id: str, org_id: str) -> None:
        """
        Delete all chunks for a document from Qdrant.

        Args:
            document_id: document to delete
            org_id: org check for safety
        """
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="org_id", match=MatchValue(value=org_id))
                ]
            )
        )

    def get_stats(self, org_id: str) -> dict:
        """Return collection stats for an org."""
        count = self.qdrant.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[FieldCondition(key="org_id", match=MatchValue(value=org_id))]
            )
        )
        return {"total_chunks": count.count, "org_id": org_id}
