from .age_tagger import AgeTagger
from .freshness_tagger import FreshnessTagger
from .usage_tracker import UsageTracker
from .weight_calculator import WeightCalculator


class TagEngine:
    """
    Orchestrates all tagging logic for a document.
    Combines age, freshness, and usage tags to produce
    a final retrieval weight.
    """

    def __init__(self, mongodb_client=None):
        self.age_tagger = AgeTagger()
        self.freshness_tagger = FreshnessTagger()
        self.usage_tracker = UsageTracker(mongodb_client)
        self.weight_calculator = WeightCalculator()
        self.mongodb_client = mongodb_client

    def tag_document(self, document: dict) -> dict:
        """
        Calculate all tags and weight for a single document.

        Args:
            document: MongoDB document dict with fields:
                      upload_timestamp, last_verified,
                      query_count, last_queried

        Returns:
            updated document dict with tags and weight added:
            {
                ...original fields,
                age_tag: str,
                freshness_tag: str,
                usage_tag: str,
                retrieval_weight: float,
                tags: list[str]  (combined list for display)
            }
        """
        upload_date = document.get("upload_timestamp")
        last_verified = document.get("last_verified")
        query_count = document.get("query_count", 0)
        last_queried = document.get("last_queried")

        # Calculate individual tags
        age_tag = self.age_tagger.calculate_age_tag(upload_date)
        freshness_tag = self.freshness_tagger.calculate_freshness_tag(last_verified)
        usage_tag = self.usage_tracker.calculate_usage_tag(query_count, last_queried)

        # Calculate weight from tag combination
        weight = self.weight_calculator.calculate_weight(
            usage_tag, age_tag, freshness_tag
        )

        # Update document dict
        document["age_tag"] = age_tag
        document["freshness_tag"] = freshness_tag
        document["usage_tag"] = usage_tag
        document["retrieval_weight"] = weight
        document["tags"] = [age_tag, freshness_tag, usage_tag]

        return document

    async def retag_all_documents(self, org_id: str) -> dict:
        """
        Retag all documents for an organisation.
        Called by tag_scheduler every 24 hours.

        Args:
            org_id: organisation ID to retag

        Returns:
            dict with total_documents and updated_count
        """
        if not self.mongodb_client:
            print("No MongoDB client available for retagging")
            return {"total_documents": 0, "updated_count": 0}

        try:
            documents = await self.mongodb_client.documents.find(
                {"org_id": org_id}
            ).to_list(length=None)

            updated_count = 0
            for document in documents:
                tagged = self.tag_document(document)
                await self.mongodb_client.documents.update_one(
                    {"_id": document["_id"]},
                    {"$set": {
                        "age_tag": tagged["age_tag"],
                        "freshness_tag": tagged["freshness_tag"],
                        "usage_tag": tagged["usage_tag"],
                        "retrieval_weight": tagged["retrieval_weight"],
                        "tags": tagged["tags"]
                    }}
                )
                updated_count += 1

            return {
                "total_documents": len(documents),
                "updated_count": updated_count
            }

        except Exception as e:
            print(f"Retagging failed: {e}")
            return {"total_documents": 0, "updated_count": 0}
