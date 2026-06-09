import time
from .retriever import Retriever
from .confidence import ConfidenceScorer
from .contradiction import ContradictionDetector
from .generator import ResponseGenerator
from ..multilingual.detector import LanguageDetector
from ..multilingual.translator import Translator
from ..tagging.usage_tracker import UsageTracker


class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrating:
    language detection → translation → retrieval → confidence →
    contradiction → generation → translation back → audit log
    """

    def __init__(self, qdrant_client, embedder, mongodb_client=None, llm_provider: str = "groq"):
        self.retriever = Retriever(qdrant_client, embedder)
        self.confidence_scorer = ConfidenceScorer()
        self.contradiction_detector = ContradictionDetector()
        self.generator = ResponseGenerator(provider=llm_provider)
        self.language_detector = LanguageDetector()
        self.translator = Translator()
        self.usage_tracker = UsageTracker(mongodb_client)
        self.mongodb_client = mongodb_client

    def run(self, query: str, user_id: str, org_id: str, user_role: str,
            session_id: str, history: list = [], document_ids: list = []) -> dict:
        """
        Run the full RAG pipeline for a user query.

        Args:
            query: user's question (any language)
            user_id: ID of the user asking
            org_id: organisation ID for data isolation
            user_role: user role for classification filter
            session_id: conversation session ID
            history: prior conversation messages for multi-turn
            document_ids: optional list to restrict to specific docs

        Returns:
            complete response dict ready to return from API
        """
        start_time = time.time()

        # Step 1 — Detect query language
        lang_result = self.language_detector.detect(query)
        original_language = lang_result["language_code"]

        # Step 2 — Translate to English if needed
        english_query = query
        if original_language != "en":
            english_query = self.translator.translate_to_english(query, original_language)

        # Step 3 — Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(
            query=english_query,
            org_id=org_id,
            user_role=user_role,
            top_k=5,
            document_ids=document_ids if document_ids else None
        )

        # Step 4 — Handle no results
        if not retrieved_chunks:
            return {
                "answer": "I could not find relevant information in your documents.",
                "confidence_score": 0.0,
                "confidence_label": "no_results",
                "confidence_warning": "No relevant documents found.",
                "has_contradiction": False,
                "contradiction_detail": None,
                "conflicting_docs": [],
                "sources": [],
                "original_language": original_language,
                "response_time_ms": int((time.time() - start_time) * 1000),
                "session_id": session_id
            }

        # Step 5 — Increment usage count for retrieved documents
        unique_doc_ids = list(set(c["document_id"] for c in retrieved_chunks))
        for doc_id in unique_doc_ids:
            # Run async usage increment in background
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self.usage_tracker.increment_query_count(doc_id, org_id))
            except Exception:
                pass

        # Step 6 — Confidence scoring
        confidence = self.confidence_scorer.calculate(retrieved_chunks)

        # Step 7 — Contradiction detection
        contradiction = self.contradiction_detector.detect(retrieved_chunks)

        # Step 8 — Generate LLM response
        answer_english = self.generator.generate(
            query=english_query,
            retrieved_chunks=retrieved_chunks,
            conversation_history=history
        )

        # Step 9 — Translate response back to original language
        final_answer = answer_english
        if original_language != "en":
            final_answer = self.translator.translate_from_english(
                answer_english, original_language
            )

        response_time = int((time.time() - start_time) * 1000)

        # Step 10 — Assemble final response
        return {
            "answer": final_answer,
            "confidence_score": confidence["score"],
            "confidence_label": confidence["label"],
            "confidence_warning": confidence.get("warning"),
            "has_contradiction": contradiction["has_contradiction"],
            "contradiction_detail": contradiction.get("contradiction_detail"),
            "conflicting_docs": contradiction.get("conflicting_docs", []),
            "sources": [
                {
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "chunk_text": chunk["chunk_text"][:200],
                    "similarity_score": chunk["similarity_score"],
                    "weighted_score": chunk["weighted_score"],
                    "age_tag": chunk.get("age_tag"),
                    "freshness_tag": chunk.get("freshness_tag")
                }
                for chunk in retrieved_chunks
            ],
            "original_language": original_language,
            "response_time_ms": response_time,
            "session_id": session_id
        }
