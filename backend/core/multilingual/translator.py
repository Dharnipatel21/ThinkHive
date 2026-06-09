try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("Warning: deep-translator not installed. Translation disabled.")


class Translator:
    """
    Translates text between languages using Google Translate
    via the deep-translator library.
    Used to support multilingual queries in the RAG pipeline.
    """

    def __init__(self):
        self.available = TRANSLATOR_AVAILABLE

    def translate_to_english(self, text: str, source_language: str) -> str:
        """
        Translate text from any language to English.

        Args:
            text: text to translate
            source_language: language code of source text (e.g. "ta", "hi")

        Returns:
            English translation or original text if already English
        """
        if not text or not text.strip():
            return text

        if source_language == "en":
            return text

        if not self.available:
            print("Translator not available. Returning original text.")
            return text

        try:
            translator = GoogleTranslator(
                source=source_language,
                target="en"
            )
            return translator.translate(text)
        except Exception as e:
            print(f"Translation to English failed: {e}")
            return text

    def translate_from_english(self, text: str, target_language: str) -> str:
        """
        Translate English text to any target language.

        Args:
            text: English text to translate
            target_language: target language code (e.g. "ta", "hi")

        Returns:
            Translated text or original if target is English
        """
        if not text or not text.strip():
            return text

        if target_language == "en":
            return text

        if not self.available:
            return text

        try:
            translator = GoogleTranslator(
                source="en",
                target=target_language
            )
            return translator.translate(text)
        except Exception as e:
            print(f"Translation from English failed: {e}")
            return text
