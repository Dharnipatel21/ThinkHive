import os
from datetime import datetime, timezone

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class MetadataExtractor:
    """
    Extracts document-level metadata — file properties, not content.
    Stored in MongoDB and used by the tagging system.
    """

    SUPPORTED_TYPES = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "doc",
        ".txt": "txt"
    }

    def __init__(self):
        pass

    def extract(self, file_path: str, extracted_text: str, uploaded_by: str) -> dict:
        """
        Extract metadata from a document file.

        Args:
            file_path: full path to uploaded file
            extracted_text: already extracted text (for language detection)
            uploaded_by: user_id of uploader

        Returns:
            metadata dict ready to store in MongoDB
        """
        filename = os.path.basename(file_path)
        extension = os.path.splitext(filename)[1].lower()
        file_type = self.SUPPORTED_TYPES.get(extension, "unknown")

        # File size in KB
        try:
            file_size_kb = round(os.path.getsize(file_path) / 1024, 2)
        except OSError:
            file_size_kb = 0.0

        # Language detection from first 1000 characters
        language = self._detect_language(extracted_text)

        # Word and character count
        word_count = len(extracted_text.split()) if extracted_text else 0
        char_count = len(extracted_text) if extracted_text else 0

        return {
            "filename": filename,
            "file_type": file_type,
            "file_size_kb": file_size_kb,
            "upload_timestamp": datetime.now(timezone.utc),
            "uploaded_by": uploaded_by,
            "language": language,
            "word_count": word_count,
            "char_count": char_count,
            "last_verified": None,
            "query_count": 0,
            "last_queried": None
        }

    def _detect_language(self, text: str) -> str:
        """Detect language from first 1000 characters of text."""
        if not text or not text.strip():
            return "en"

        if not LANGDETECT_AVAILABLE:
            return "en"

        try:
            sample = text[:1000].strip()
            if len(sample) < 20:
                return "en"
            return detect(sample)
        except Exception:
            return "en"
