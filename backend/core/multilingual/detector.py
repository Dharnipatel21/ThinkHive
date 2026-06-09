try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

LANGUAGE_NAMES = {
    "en": "English", "ta": "Tamil", "hi": "Hindi",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi",
    "fr": "French", "de": "German", "es": "Spanish",
    "zh-cn": "Chinese", "ja": "Japanese", "ar": "Arabic"
}


class LanguageDetector:
    """
    Detects the language of input text using langdetect.
    Used before translation to determine if translation is needed.
    """

    def __init__(self):
        pass

    def detect(self, text: str) -> dict:
        """
        Detect language of text.

        Args:
            text: input text to detect

        Returns:
            dict with language_code and language_name
        """
        if not text or not text.strip():
            return {"language_code": "en", "language_name": "English"}

        if not LANGDETECT_AVAILABLE:
            return {"language_code": "en", "language_name": "English"}

        try:
            # Use first 500 characters for faster detection
            sample = text[:500].strip()
            if len(sample) < 10:
                return {"language_code": "en", "language_name": "English"}

            lang_code = detect(sample)
            lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())

            return {
                "language_code": lang_code,
                "language_name": lang_name
            }

        except Exception:
            return {"language_code": "en", "language_name": "English"}

    def is_english(self, text: str) -> bool:
        """Check if text is English."""
        result = self.detect(text)
        return result["language_code"] == "en"
