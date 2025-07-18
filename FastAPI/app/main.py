import asyncio
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow
import os
import pandas as pd

from app.routers import home
from app.scheduler import background_scheduler, model_updater_scheduler
from app.services import calculate_and_set_all_metrics, load_and_cache_model
from app.config import DATA_DIR, MLFLOW_TRACKING_URI

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    app = FastAPI(title="Model Monitoring Service")

    Instrumentator().instrument(app).expose(app)

    app.include_router(home.router)

    @app.on_event("startup")
    async def startup_event():
        """On app startup, perform all initialization tasks."""
        print("--- Server starting up: Loading initial model for metrics calculation... ---")
        await load_and_cache_model()
        
        print("--- Calculating initial metrics... ---")
        if not os.path.exists(os.path.join(DATA_DIR, 'data.csv')):
            os.makedirs(DATA_DIR, exist_ok=True)
            pd.DataFrame({
                'Review': ['good', 'bad', 'great', 'terrible', 'ok'],
                'Sentiment': [1, 0, 1, 0, 1]
            }).to_csv(os.path.join(DATA_DIR, 'data.csv'), index=False)
        
        await calculate_and_set_all_metrics()

        print("--- Starting background schedulers... ---")
        asyncio.create_task(background_scheduler())
        asyncio.create_task(model_updater_scheduler())

    return app

app = create_app()
