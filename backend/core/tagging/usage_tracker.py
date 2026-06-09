from datetime import datetime, timezone


class UsageTracker:
    """
    Tracks how frequently each document is queried.
    Usage data is stored in MongoDB and used for usage tags.
    """

    def __init__(self, mongodb_client=None):
        """
        Args:
            mongodb_client: async MongoDB client from Member 3.
                            Can be None for testing - falls back to in-memory.
        """
        self.mongodb_client = mongodb_client
        self._in_memory_counts = {}  # fallback for testing

    async def increment_query_count(self, document_id: str, org_id: str) -> None:
        """
        Increment the query count for a document in MongoDB.

        Args:
            document_id: MongoDB document ID
            org_id: organisation ID for isolation
        """
        if self.mongodb_client:
            try:
                await self.mongodb_client.documents.update_one(
                    {"_id": document_id, "org_id": org_id},
                    {
                        "$inc": {"query_count": 1},
                        "$set": {"last_queried": datetime.now(timezone.utc)}
                    }
                )
            except Exception as e:
                print(f"Failed to increment query count: {e}")
        else:
            # Fallback for testing
            self._in_memory_counts[document_id] = (
                self._in_memory_counts.get(document_id, 0) + 1
            )

    def calculate_usage_tag(self, query_count: int, last_queried: datetime) -> str:
        """
        Calculate usage tag from query count and recency.

        Args:
            query_count: total number of times document was queried
            last_queried: datetime of last query. None = never queried.

        Returns:
            one of: frequently_used, active, least_used, unused
        """
        if not last_queried or query_count == 0:
            return "unused"

        now = datetime.now(timezone.utc)
        if last_queried.tzinfo is None:
            last_queried = last_queried.replace(tzinfo=timezone.utc)

        days_since_queried = (now - last_queried).days

        if query_count >= 20:
            return "frequently_used"
        elif days_since_queried <= 7:
            return "active"
        elif query_count < 3:
            return "least_used"
        elif days_since_queried > 30:
            return "unused"
        else:
            return "active"
