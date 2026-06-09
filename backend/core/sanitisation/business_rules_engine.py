import re
import os
import yaml


class BusinessRulesEngine:
    """
    Applies organisation-specific custom sanitisation rules
    loaded from YAML config files and MongoDB custom rules.
    """

    RULES_DIR = os.path.join(os.path.dirname(__file__), "rules")

    def __init__(self, domain: str, org_custom_rules: list = []):
        """
        Args:
            domain: one of tech, manufacturing, ecommerce
            org_custom_rules: list of custom rule dicts from MongoDB
        """
        self.domain = domain
        self.rules = []
        self._load_domain_rules(domain)
        self._load_custom_rules(org_custom_rules)

    def _load_domain_rules(self, domain: str):
        """Load rules from domain YAML file."""
        rules_file = os.path.join(self.RULES_DIR, f"{domain}_rules.yaml")
        if not os.path.exists(rules_file):
            print(f"No rules file found for domain: {domain}")
            return

        try:
            with open(rules_file, "r") as f:
                data = yaml.safe_load(f)
                if data and "rules" in data:
                    self.rules.extend(data["rules"])
        except Exception as e:
            print(f"Error loading rules file: {e}")

    def _load_custom_rules(self, org_custom_rules: list):
        """Load custom rules from MongoDB org config."""
        if org_custom_rules:
            self.rules.extend(org_custom_rules)

    def apply_rules(self, text: str) -> dict:
        """
        Apply all loaded rules to the text.

        Args:
            text: document text after PII and salary masking

        Returns:
            dict with masked_text and rules_applied list
        """
        if not text or not text.strip():
            return {"masked_text": text, "rules_applied": []}

        masked_text = text
        rules_applied = []

        for rule in self.rules:
            rule_name = rule.get("name", "unnamed")
            pattern_type = rule.get("pattern_type", "keyword")
            pattern = rule.get("pattern", "")
            replacement = rule.get("replacement", "[REDACTED]")

            if not pattern:
                continue

            try:
                if pattern_type == "regex":
                    compiled = re.compile(pattern, re.IGNORECASE)
                    if compiled.search(masked_text):
                        masked_text = compiled.sub(replacement, masked_text)
                        rules_applied.append(rule_name)

                elif pattern_type == "keyword":
                    if pattern.lower() in masked_text.lower():
                        masked_text = re.sub(
                            re.escape(pattern),
                            replacement,
                            masked_text,
                            flags=re.IGNORECASE
                        )
                        rules_applied.append(rule_name)

            except Exception as e:
                print(f"Error applying rule '{rule_name}': {e}")
                continue

        return {
            "masked_text": masked_text,
            "rules_applied": rules_applied
        }
