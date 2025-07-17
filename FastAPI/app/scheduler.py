import asyncio
from app.services import calculate_and_set_all_metrics, load_and_cache_model

async def background_scheduler():
    """Runs in the background, updating metrics every 60 seconds."""
    while True:
        await asyncio.sleep(60)
        print("--- Running scheduled metrics update ---")
        await calculate_and_set_all_metrics()

async def model_updater_scheduler():
    """
    Periodically checks for a new model version and triggers a reload.
    This is our "hot-swapping" mechanism.
    """
    while True:
        await asyncio.sleep(300) # Check for a new model every 5 minutes
        print("--- Running scheduled model update check ---")
        try:
            await load_and_cache_model()
        except Exception as e:
            print(f"Scheduled model update failed: {e}")
