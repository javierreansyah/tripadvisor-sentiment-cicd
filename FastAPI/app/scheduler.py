# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

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
        await asyncio.sleep(60) # Check for a new model every 5 minutes
        print("--- Running scheduled model update check ---")
        try:
            await load_and_cache_model()
        except Exception as e:
            print(f"Scheduled model update failed: {e}")
