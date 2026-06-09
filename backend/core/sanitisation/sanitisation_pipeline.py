from datetime import datetime, timezone
from .pii_detector import PIIDetector
from .signature_detector import SignatureDetector
from .salary_detector import SalaryDetector
from .business_rules_engine import BusinessRulesEngine


class SanitisationPipeline:
    """
    Orchestrates all sanitisation steps in order:
    1. PII Detection and masking
    2. Signature detection and removal
    3. Salary and compensation masking
    4. Custom business rules application

    Produces a sanitised copy of the text and a full
    sanitisation log showing what was redacted and why.
    """

    def __init__(self, domain: str, org_custom_rules: list = []):
        """
        Args:
            domain: tech, manufacturing, or ecommerce
            org_custom_rules: custom rules from org MongoDB config
        """
        self.pii_detector = PIIDetector()
        self.signature_detector = SignatureDetector()
        self.salary_detector = SalaryDetector()
        self.business_rules = BusinessRulesEngine(domain, org_custom_rules)

    def run(self, text: str, language: str = "en", image_paths: list = []) -> dict:
        """
        Run full sanitisation pipeline on document text.

        Args:
            text: raw extracted text from document
            language: detected document language
            image_paths: list of image file paths for signature detection

        Returns:
            dict with:
            {
                sanitised_text: str,
                sanitisation_log: {
                    pii: { entity_count, has_pii, entities },
                    signatures: { detected, count },
                    salary: { redaction_count, redacted_items },
                    business_rules: { rules_applied },
                    total_redactions: int,
                    timestamp: datetime
                }
            }
        """
        sanitisation_log = {
            "pii": {},
            "signatures": {},
            "salary": {},
            "business_rules": {},
            "total_redactions": 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        current_text = text

        # Step 1 — PII Detection
        pii_result = self.pii_detector.detect_and_mask(current_text, language)
        current_text = pii_result["masked_text"]
        sanitisation_log["pii"] = {
            "entity_count": pii_result["entity_count"],
            "has_pii": pii_result["has_pii"],
            "entities": pii_result["detected_entities"]
        }

        # Step 2 — Signature Detection (image-based)
        sig_results = []
        for image_path in image_paths:
            sig_result = self.signature_detector.detect_signatures(image_path)
            if sig_result["has_signature"]:
                cleaned_path = image_path.replace(".", "_clean.")
                self.signature_detector.remove_signatures(image_path, cleaned_path)
                sig_results.append({
                    "image": image_path,
                    "regions": sig_result["regions"]
                })

        sanitisation_log["signatures"] = {
            "detected": len(sig_results) > 0,
            "count": len(sig_results),
            "details": sig_results
        }

        # Step 3 — Salary Detection
        salary_result = self.salary_detector.detect_and_mask(current_text)
        current_text = salary_result["masked_text"]
        sanitisation_log["salary"] = {
            "redaction_count": salary_result["redaction_count"],
            "redacted_items": salary_result["redacted_items"]
        }

        # Step 4 — Business Rules
        rules_result = self.business_rules.apply_rules(current_text)
        current_text = rules_result["masked_text"]
        sanitisation_log["business_rules"] = {
            "rules_applied": rules_result["rules_applied"]
        }

        # Total redaction count
        sanitisation_log["total_redactions"] = (
            pii_result["entity_count"] +
            len(sig_results) +
            salary_result["redaction_count"] +
            len(rules_result["rules_applied"])
        )

        return {
            "sanitised_text": current_text,
            "sanitisation_log": sanitisation_log
        }
