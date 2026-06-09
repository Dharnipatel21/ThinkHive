from ..rag.retriever import Retriever
from ..rag.generator import ResponseGenerator


class Summariser:
    """
    Generates summaries of single or multiple documents using
    a map-reduce approach to handle long documents that exceed
    the LLM context window.

    Map step: summarise each chunk individually
    Reduce step: combine all chunk summaries into one final summary
    """

    def __init__(self, retriever: Retriever, generator: ResponseGenerator):
        self.retriever = retriever
        self.generator = generator

    def summarise_document(self, document_id: str, org_id: str,
                           user_role: str = "employee") -> str:
        """
        Summarise a single document using map-reduce.

        Args:
            document_id: MongoDB document ID to summarise
            org_id: organisation ID
            user_role: user role for access control

        Returns:
            final summary string
        """
        # Retrieve all chunks for this document
        chunks = self.retriever.retrieve(
            query="summarise this document",
            org_id=org_id,
            user_role=user_role,
            top_k=20,
            document_ids=[document_id]
        )

        if not chunks:
            return "No content available to summarise for this document."

        return self._map_reduce(chunks)

    def summarise_multi_document(self, query: str, org_id: str,
                                  user_role: str = "employee") -> str:
        """
        Summarise across multiple relevant documents.

        Args:
            query: topic or question to focus the summary on
            org_id: organisation ID
            user_role: user role for access control

        Returns:
            unified summary string
        """
        chunks = self.retriever.retrieve(
            query=query,
            org_id=org_id,
            user_role=user_role,
            top_k=15
        )

        if not chunks:
            return "No relevant documents found for this summary."

        return self._map_reduce(chunks)

    def generate_report(self, query: str, org_id: str,
                        user_role: str, report_title: str) -> dict:
        """
        Generate a structured report from multiple documents.

        Args:
            query: what the report should cover
            org_id: organisation ID
            user_role: user role for access control
            report_title: title for the report

        Returns:
            dict with title, summary, sources, and section hints
        """
        chunks = self.retriever.retrieve(
            query=query,
            org_id=org_id,
            user_role=user_role,
            top_k=15
        )

        if not chunks:
            return {
                "title": report_title,
                "summary": "No relevant content found for this report.",
                "sources": [],
                "generated_at": None
            }

        summary = self._map_reduce(chunks, style="report")
        sources = list(set(
            f"{c['document_name']} (p.{c['page_number']})"
            for c in chunks
        ))

        from datetime import datetime, timezone
        return {
            "title": report_title,
            "summary": summary,
            "sources": sources,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    def _map_reduce(self, chunks: list[dict], style: str = "summary") -> str:
        """
        Run map-reduce summarisation on a list of chunks.

        Map: summarise each chunk individually (batched in groups of 5)
        Reduce: combine all mini summaries into one final summary

        Args:
            chunks: list of chunk dicts
            style: "summary" or "report"

        Returns:
            final summarised text
        """
        if not chunks:
            return "No content to summarise."

        # MAP STEP - summarise in batches of 5 chunks
        batch_size = 5
        mini_summaries = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            map_prompt = "Summarise the following passages in 3-5 sentences, preserving key facts:"
            mini_summary = self.generator.generate(
                query=map_prompt,
                retrieved_chunks=batch,
                conversation_history=[]
            )
            mini_summaries.append(mini_summary)

        if len(mini_summaries) == 1:
            return mini_summaries[0]

        # REDUCE STEP - combine all mini summaries
        combined_text = "\n\n".join([
            f"Section {i+1}:\n{summary}"
            for i, summary in enumerate(mini_summaries)
        ])

        if style == "report":
            reduce_query = "Combine these sections into a structured executive summary with key findings and recommendations:"
        else:
            reduce_query = "Combine these sections into one coherent, concise summary:"

        # Create fake chunks for the reduce step
        reduce_chunks = [{
            "chunk_text": combined_text,
            "document_name": "Combined Summaries",
            "page_number": 1,
            "similarity_score": 1.0,
            "weighted_score": 1.0,
            "freshness_tag": "fresh",
            "age_tag": "new"
        }]

        final_summary = self.generator.generate(
            query=reduce_query,
            retrieved_chunks=reduce_chunks,
            conversation_history=[]
        )

        return final_summary
