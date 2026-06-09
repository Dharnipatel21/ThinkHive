from datetime import datetime, timezone


class FreshnessTagger:
    """
    Tags documents based on when they were last verified
    by a human. Freshness tags affect retrieval weight.
    """

    def __init__(self):
        # Thresholds in days
        self.fresh_threshold = 30
        self.stale_threshold = 60
        # Beyond stale_threshold = expired

    def calculate_freshness_tag(self, last_verified: datetime) -> str:
        """
        Calculate freshness tag based on last verification date.

        Args:
            last_verified: datetime when document was last verified.
                           None means never verified.

        Returns:
            one of: fresh, stale, expired, unverified
        """
        if not last_verified:
            return "unverified"

        now = datetime.now(timezone.utc)

        if last_verified.tzinfo is None:
            last_verified = last_verified.replace(tzinfo=timezone.utc)

        days_since_verified = (now - last_verified).days

        if days_since_verified <= self.fresh_threshold:
            return "fresh"
        elif days_since_verified <= self.stale_threshold:
            return "stale"
        else:
            return "expired"
