class ConfidenceScorer:
    """
    Calculates confidence score for a RAG response based on
    how well the retrieved chunks matched the query.
    """

    def __init__(self):
        self.high_threshold = 0.85
        self.medium_threshold = 0.70
        # Weights for top-5 chunks (top chunk matters most)
        self.weights = [0.40, 0.25, 0.15, 0.12, 0.08]

    def calculate(self, retrieved_chunks: list[dict]) -> dict:
        """
        Calculate confidence from retrieved chunk scores.

        Args:
            retrieved_chunks: list of chunk dicts with weighted_score key

        Returns:
            dict with score, label, color, warning
        """
        if not retrieved_chunks:
            return {
                "score": 0.0,
                "label": "no_results",
                "color": "gray",
                "warning": "No relevant documents found in the knowledge base."
            }

        scores = [chunk.get("weighted_score", 0.0) for chunk in retrieved_chunks]

        # Weighted average — top chunk contributes most
        total_weight = 0
        weighted_sum = 0
        for i, score in enumerate(scores[:len(self.weights)]):
            w = self.weights[i]
            weighted_sum += score * w
            total_weight += w

        final_score = round(weighted_sum / total_weight if total_weight > 0 else 0.0, 3)

        # Map to label
        if final_score >= self.high_threshold:
            label = "high"
            color = "green"
            warning = None
        elif final_score >= self.medium_threshold:
            label = "medium"
            color = "amber"
            warning = "Answer is moderately grounded. Verify with source documents."
        else:
            label = "low"
            color = "red"
            warning = "Low confidence. Answer may not be fully grounded in your documents."

        # Check if any source documents are outdated
        outdated_docs = [
            c["document_name"] for c in retrieved_chunks
            if c.get("freshness_tag") in ["expired", "stale"]
               or c.get("age_tag") == "outdated"
        ]

        if outdated_docs:
            outdated_warning = f"Note: Answer includes content from potentially outdated documents: {', '.join(set(outdated_docs))}. Please verify."
            warning = outdated_warning if not warning else f"{warning} {outdated_warning}"

        return {
            "score": final_score,
            "label": label,
            "color": color,
            "warning": warning
        }
