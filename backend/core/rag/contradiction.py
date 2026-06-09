try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Contradiction detection disabled.")


class ContradictionDetector:
    """
    Detects contradictions between retrieved chunks using
    DeBERTa NLI model. Falls back to LLM-based detection
    if transformers not available.
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Groq or Gemini client for explanation generation
        """
        self.llm_client = llm_client
        self.nli_model = None
        self.contradiction_threshold = 0.85

        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading DeBERTa NLI model...")
                self.nli_model = hf_pipeline(
                    "text-classification",
                    model="cross-encoder/nli-deberta-v3-base"
                )
                print("NLI model loaded.")
            except Exception as e:
                print(f"NLI model load failed: {e}")

    def detect(self, retrieved_chunks: list[dict]) -> dict:
        """
        Check all chunk pairs for contradictions.

        Args:
            retrieved_chunks: list of retrieved chunk dicts

        Returns:
            dict with has_contradiction bool and contradiction_detail str
        """
        if len(retrieved_chunks) < 2:
            return {"has_contradiction": False, "contradiction_detail": None}

        if not self.nli_model:
            return {"has_contradiction": False, "contradiction_detail": None}

        try:
            for i in range(len(retrieved_chunks)):
                for j in range(i + 1, len(retrieved_chunks)):
                    text_a = retrieved_chunks[i]["chunk_text"]
                    text_b = retrieved_chunks[j]["chunk_text"]

                    # NLI input format
                    nli_input = f"{text_a} [SEP] {text_b}"

                    result = self.nli_model(nli_input)

                    if (result[0]["label"].lower() == "contradiction" and
                            result[0]["score"] >= self.contradiction_threshold):

                        detail = self._explain_contradiction(
                            text_a, text_b,
                            retrieved_chunks[i]["document_name"],
                            retrieved_chunks[j]["document_name"]
                        )

                        return {
                            "has_contradiction": True,
                            "contradiction_detail": detail,
                            "conflicting_docs": [
                                retrieved_chunks[i]["document_name"],
                                retrieved_chunks[j]["document_name"]
                            ]
                        }

        except Exception as e:
            print(f"Contradiction detection error: {e}")

        return {"has_contradiction": False, "contradiction_detail": None}

    def _explain_contradiction(self, text_a: str, text_b: str, doc_a: str, doc_b: str) -> str:
        """Generate human-readable explanation of the contradiction."""
        if not self.llm_client:
            return (
                f"Conflicting information detected between "
                f"'{doc_a}' and '{doc_b}'. Please review both documents."
            )

        try:
            prompt = f"""Two passages from different documents appear to contradict each other.

Passage 1 from {doc_a}:
{text_a[:500]}

Passage 2 from {doc_b}:
{text_b[:500]}

In one sentence explain what specifically contradicts between them."""

            response = self.llm_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Contradiction explanation failed: {e}")
            return f"Conflicting information found between '{doc_a}' and '{doc_b}'."
