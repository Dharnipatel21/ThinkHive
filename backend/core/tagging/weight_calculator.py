class WeightCalculator:
    """
    Calculates retrieval weight for a document based on its tags.
    Weight multiplies cosine similarity score during retrieval.
    Higher weight = document ranks higher in search results.
    """

    def __init__(self):
        # Weight map: (usage_tag, age_tag, freshness_tag) -> weight
        self.weight_map = {
            # Best case - active, new, fresh
            ("frequently_used", "new", "fresh"):     1.0,
            ("frequently_used", "recent", "fresh"):  1.0,
            ("active", "new", "fresh"):               0.95,
            ("active", "recent", "fresh"):            0.90,

            # Good - active but aging
            ("frequently_used", "new", "stale"):     0.85,
            ("frequently_used", "recent", "stale"):  0.85,
            ("active", "new", "stale"):               0.80,
            ("active", "recent", "stale"):            0.75,
            ("active", "old", "fresh"):               0.70,

            # Unverified new docs - decent but unconfirmed
            ("active", "new", "unverified"):          0.75,
            ("active", "recent", "unverified"):       0.65,

            # Least used
            ("least_used", "recent", "fresh"):        0.60,
            ("least_used", "new", "fresh"):           0.65,
            ("least_used", "old", "fresh"):           0.50,
            ("least_used", "recent", "stale"):        0.45,

            # Unused
            ("unused", "recent", "fresh"):            0.40,
            ("unused", "old", "stale"):               0.30,
            ("unused", "outdated", "stale"):          0.20,

            # Expired - lowest weight
            ("unused", "outdated", "expired"):        0.10,
            ("least_used", "outdated", "expired"):    0.10,
            ("active", "outdated", "expired"):        0.15,
            ("frequently_used", "outdated", "expired"): 0.20,
        }

        # Default weights by freshness if exact combo not found
        self.freshness_defaults = {
            "fresh": 0.70,
            "stale": 0.50,
            "unverified": 0.60,
            "expired": 0.15
        }

    def calculate_weight(self, usage_tag: str, age_tag: str, freshness_tag: str) -> float:
        """
        Calculate retrieval weight for a document.

        Args:
            usage_tag: frequently_used, active, least_used, unused
            age_tag: new, recent, old, outdated
            freshness_tag: fresh, stale, expired, unverified

        Returns:
            float between 0.1 and 1.0
        """
        key = (usage_tag, age_tag, freshness_tag)

        # Direct lookup
        if key in self.weight_map:
            return self.weight_map[key]

        # Expired override - always low weight
        if freshness_tag == "expired":
            return 0.15

        # Fallback by freshness
        return self.freshness_defaults.get(freshness_tag, 0.50)

    def get_weight_map(self) -> dict:
        """Return full weight map for admin display."""
        return {str(k): v for k, v in self.weight_map.items()}
