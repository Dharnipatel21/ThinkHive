class DocumentClassifier:
    """
    Classifies documents into sensitivity levels:
    public, internal, restricted, confidential.
    Used for access control in retrieval.
    """

    VALID_LEVELS = ["public", "internal", "restricted", "confidential"]

    def __init__(self):
        self.confidential_keywords = [
            "strictly confidential", "highly confidential",
            "confidential", "do not distribute", "do not share",
            "for your eyes only", "top secret", "sensitive"
        ]
        self.restricted_keywords = [
            "restricted", "internal use only", "not for public",
            "limited distribution", "private", "not for release",
            "management only", "hr only", "finance only"
        ]
        self.public_keywords = [
            "public release", "press release", "for immediate release",
            "public document", "publicly available", "open to all"
        ]

    def classify(self, text: str, filename: str, user_input: str = None) -> str:
        """
        Determine classification level of a document.

        Args:
            text: extracted document text
            filename: original filename
            user_input: manually set classification (takes priority)

        Returns:
            one of: public, internal, restricted, confidential
        """
        # User manual input takes highest priority
        if user_input and user_input.lower() in self.VALID_LEVELS:
            return user_input.lower()

        # Scan first 500 chars of text and full filename
        scan_text = (text[:500] + " " + filename).lower()

        # Check from highest to lowest sensitivity
        for keyword in self.confidential_keywords:
            if keyword in scan_text:
                return "confidential"

        for keyword in self.restricted_keywords:
            if keyword in scan_text:
                return "restricted"

        for keyword in self.public_keywords:
            if keyword in scan_text:
                return "public"

        # Default to internal
        return "internal"
