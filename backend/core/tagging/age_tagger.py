from datetime import datetime, timezone


class AgeTagger:
    """
    Tags documents based on how long ago they were uploaded.
    Age tags influence retrieval weight - newer docs rank higher.
    """

    def __init__(self):
        # Thresholds in days
        self.new_threshold = 7
        self.recent_threshold = 30
        self.old_threshold = 90
        # Beyond old_threshold = outdated

    def calculate_age_tag(self, upload_date: datetime) -> str:
        """
        Calculate age tag based on upload date.

        Args:
            upload_date: datetime when document was uploaded

        Returns:
            one of: new, recent, old, outdated
        """
        if not upload_date:
            return "unknown"

        now = datetime.now(timezone.utc)

        # Make upload_date timezone-aware if it is not
        if upload_date.tzinfo is None:
            upload_date = upload_date.replace(tzinfo=timezone.utc)

        days_old = (now - upload_date).days

        if days_old <= self.new_threshold:
            return "new"
        elif days_old <= self.recent_threshold:
            return "recent"
        elif days_old <= self.old_threshold:
            return "old"
        else:
            return "outdated"

    def get_days_old(self, upload_date: datetime) -> int:
        """Return number of days since upload."""
        if not upload_date:
            return 0
        now = datetime.now(timezone.utc)
        if upload_date.tzinfo is None:
            upload_date = upload_date.replace(tzinfo=timezone.utc)
        return (now - upload_date).days
