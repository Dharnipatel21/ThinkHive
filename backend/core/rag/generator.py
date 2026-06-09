try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

import os

SYSTEM_PROMPT = """You are an enterprise knowledge assistant for an organisation.

Follow these rules strictly:
1. Answer ONLY using the provided context passages
2. After each factual claim cite the source as [Document Name, Page X]
3. If the answer is not found in the context respond with exactly:
   "I could not find this information in the available documents."
4. Never make up information or use outside knowledge
5. If context passages contradict each other mention the conflict
6. Keep answers clear, concise, and professional
7. If a document is flagged as outdated mention it in your answer"""


class ResponseGenerator:
    """
    Generates grounded LLM responses using retrieved chunks as context.
    Supports Groq (LLaMA 3) and Google Gemini as free LLM providers.
    """

    def __init__(self, provider: str = "groq"):
        """
        Args:
            provider: "groq" or "gemini"
        """
        self.provider = provider
        self.client = None
        self._initialise_client()

    def _initialise_client(self):
        """Initialise the LLM client based on provider."""
        if self.provider == "groq" and GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.client = Groq(api_key=api_key)
            else:
                print("GROQ_API_KEY not set in environment")

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai.GenerativeModel("gemini-1.5-flash")
                else:
                    print("GOOGLE_API_KEY not set in environment")
            except ImportError:
                print("google-generativeai not installed")

    def generate(self, query: str, retrieved_chunks: list[dict], conversation_history: list = []) -> str:
        """
        Generate a response grounded in retrieved chunks.

        Args:
            query: user's question
            retrieved_chunks: list of relevant chunk dicts
            conversation_history: list of prior messages for multi-turn

        Returns:
            answer string with inline citations
        """
        if not retrieved_chunks:
            return "I could not find relevant information in your documents."

        # Build context from chunks
        context_parts = []
        for chunk in retrieved_chunks:
            freshness_warning = ""
            if chunk.get("freshness_tag") in ["expired", "stale"]:
                freshness_warning = " [OUTDATED - verify this information]"
            if chunk.get("age_tag") == "outdated":
                freshness_warning = " [OUTDATED - verify this information]"

            context_parts.append(
                f"[{chunk['document_name']}, Page {chunk['page_number']}]{freshness_warning}:\n{chunk['chunk_text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        user_message = f"""Context passages from the knowledge base:

{context}

---

Question: {query}

Answer based only on the above context. Include citations."""

        if self.provider == "groq" and self.client:
            return self._generate_groq(user_message, conversation_history)
        elif self.provider == "gemini" and self.client:
            return self._generate_gemini(user_message)
        else:
            return self._fallback_response(retrieved_chunks, query)

    def _generate_groq(self, user_message: str, history: list) -> str:
        """Generate using Groq API."""
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Groq generation error: {e}")
            return "An error occurred while generating the response."

    def _generate_gemini(self, user_message: str) -> str:
        """Generate using Google Gemini API."""
        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}"
            response = self.client.generate_content(full_prompt)
            return response.text

        except Exception as e:
            print(f"Gemini generation error: {e}")
            return "An error occurred while generating the response."

    def _fallback_response(self, chunks: list[dict], query: str) -> str:
        """Simple fallback when no LLM client available."""
        if chunks:
            top = chunks[0]
            return (
                f"Based on [{top['document_name']}, Page {top['page_number']}]:\n\n"
                f"{top['chunk_text']}\n\n"
                f"(Note: LLM client not configured. Showing raw retrieved content.)"
            )
        return "No relevant information found."
