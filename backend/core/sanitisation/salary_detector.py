import re


class SalaryDetector:
    """
    Detects and masks salary and compensation figures.
    Separate from PII detection because salary figures
    are numbers without obvious PII context markers.
    """

    def __init__(self):
        self.salary_keywords = [
            "salary", "ctc", "cost to company", "compensation",
            "bonus", "incentive", "remuneration", "stipend",
            "package", "hike", "increment", "gross", "net pay",
            "take home", "per annum", "per month", "lpa",
            "lakh", "annual pay", "monthly pay", "fixed pay",
            "variable pay", "total pay", "offer letter"
        ]

        # Currency patterns
        self.salary_patterns = [
            # Indian rupee symbol with numbers: ₹12,00,000 or ₹1200000
            re.compile(r'₹\s*[\d,]+(\.\d+)?', re.IGNORECASE),
            # Rs. format: Rs. 85,000 or Rs 85000
            re.compile(r'Rs\.?\s*[\d,]+(\.\d+)?', re.IGNORECASE),
            # INR format: INR 1,20,000
            re.compile(r'INR\s*[\d,]+(\.\d+)?', re.IGNORECASE),
            # LPA format: 12 LPA or 12.5 LPA
            re.compile(r'\d+(\.\d+)?\s*LPA', re.IGNORECASE),
            # Lakh format: 12 lakh or 12.5 lakhs
            re.compile(r'\d+(\.\d+)?\s*lakh(s)?', re.IGNORECASE),
            # Number followed by /month or /year
            re.compile(r'[\d,]+(\.\d+)?\s*/\s*(month|year|annum)', re.IGNORECASE),
        ]

    def detect_and_mask(self, text: str) -> dict:
        """
        Find and mask salary figures in text.

        Args:
            text: document text after PII masking

        Returns:
            dict with masked_text, redaction_count, redacted_items
        """
        if not text or not text.strip():
            return {
                "masked_text": text,
                "redaction_count": 0,
                "redacted_items": []
            }

        masked_text = text
        redacted_items = []

        # Apply direct currency regex patterns across full text
        for pattern in self.salary_patterns:
            matches = pattern.findall(masked_text)
            for match in matches:
                original = match if isinstance(match, str) else match[0]
                redacted_items.append(original)

            masked_text = pattern.sub("[SALARY_REDACTED]", masked_text)

        # Context-based masking around salary keywords
        text_lower = masked_text.lower()
        for keyword in self.salary_keywords:
            pos = 0
            while True:
                idx = text_lower.find(keyword, pos)
                if idx == -1:
                    break

                # Extract context window around keyword
                start = max(0, idx - 50)
                end = min(len(masked_text), idx + len(keyword) + 100)
                context = masked_text[start:end]

                # Find standalone numbers in context not already masked
                number_pattern = re.compile(r'\b[\d,]+(\.\d+)?\b')
                context_masked = number_pattern.sub("[SALARY_REDACTED]", context)

                # Replace context in full text
                masked_text = masked_text[:start] + context_masked + masked_text[end:]
                text_lower = masked_text.lower()
                pos = idx + len(keyword)

        # Remove duplicate redaction markers
        masked_text = re.sub(
            r'\[SALARY_REDACTED\](\s*\[SALARY_REDACTED\])+',
            '[SALARY_REDACTED]',
            masked_text
        )

        return {
            "masked_text": masked_text,
            "redaction_count": len(redacted_items),
            "redacted_items": redacted_items
        }
