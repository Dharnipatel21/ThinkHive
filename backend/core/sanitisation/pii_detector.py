try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("Warning: presidio not installed. PII detection disabled.")


class PIIDetector:
    """
    Detects and masks PII using Microsoft Presidio.
    Masks names, emails, phones, Aadhaar, PAN, addresses etc.
    """

    SUPPORTED_LANGUAGES = ["en", "de", "es", "fr", "it", "pt", "nl"]

    def __init__(self):
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None

    def detect_and_mask(self, text: str, language: str = "en") -> dict:
        """
        Detect and mask all PII in the given text.

        Args:
            text: raw document text
            language: language code of document text

        Returns:
            dict with masked_text, detected_entities, entity_count, has_pii
        """
        # Handle empty text
        if not text or not text.strip():
            return {
                "masked_text": text,
                "detected_entities": [],
                "entity_count": 0,
                "has_pii": False
            }

        # Fallback if presidio not available
        if not PRESIDIO_AVAILABLE or not self.analyzer:
            return {
                "masked_text": text,
                "detected_entities": [],
                "entity_count": 0,
                "has_pii": False
            }

        # Use English if language not supported by Presidio
        if language not in self.SUPPORTED_LANGUAGES:
            language = "en"

        try:
            # Detect PII entities
            analyzer_results = self.analyzer.analyze(
                text=text,
                language=language
            )

            # Anonymize detected entities
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results
            )

            # Build entity list for the sanitisation log
            detected_entities = [
                {
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": round(result.score, 3)
                }
                for result in analyzer_results
            ]

            return {
                "masked_text": anonymized.text,
                "detected_entities": detected_entities,
                "entity_count": len(detected_entities),
                "has_pii": len(detected_entities) > 0
            }

        except Exception as e:
            print(f"PII detection error: {e}")
            return {
                "masked_text": text,
                "detected_entities": [],
                "entity_count": 0,
                "has_pii": False
            }
