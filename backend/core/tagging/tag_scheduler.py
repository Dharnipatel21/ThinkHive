from celery import Celery
from celery.schedules import crontab


# This file sets up the Celery beat schedule for automatic retagging.
# The actual Celery app is initialised by Member 3 in tasks/celery_app.py
# This scheduler hooks into that app.

class TagScheduler:
    """
    Schedules automatic document retagging using Celery beat.
    Runs every 24 hours at midnight to refresh all document tags.
    """

    def __init__(self, celery_app: Celery = None):
        self.celery_app = celery_app

    def register_schedule(self):
        """
        Register the daily retag task with Celery beat.
        Call this during app startup.
        """
        if not self.celery_app:
            print("No Celery app provided to TagScheduler")
            return

        self.celery_app.conf.beat_schedule.update({
            "daily-document-retag": {
                "task": "tasks.tag_update_task.retag_all_orgs",
                "schedule": crontab(hour=0, minute=0),  # midnight daily
                "args": []
            }
        })
        print("Tag scheduler registered - runs daily at midnight")


# Standalone Celery task function
# Imported by tasks/tag_update_task.py
async def retag_org_documents(org_id: str, mongodb_client) -> dict:
    """
    Retag all documents for a specific organisation.
    Called by the Celery worker.

    Args:
        org_id: organisation to retag
        mongodb_client: active MongoDB client

    Returns:
        result dict with counts
    """
    from .tag_engine import TagEngine

    engine = TagEngine(mongodb_client)
    result = await engine.retag_all_documents(org_id)
    print(f"Retagged org {org_id}: {result}")
    return result
