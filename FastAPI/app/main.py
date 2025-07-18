import asyncio
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow
import os
import pandas as pd

# Import routers and services
from app.routers import home, data, dashboard
from app.scheduler import background_scheduler, model_updater_scheduler
from app.services import calculate_and_set_all_metrics, load_and_cache_model
from app.config import DATA_DIR, MLFLOW_TRACKING_URI

# --- APP FACTORY ---
def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    # Set up MLflow tracking URI for this app
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    app = FastAPI(title="Model Monitoring Service")

    # Expose Prometheus metrics
    Instrumentator().instrument(app).expose(app)

    # Include the necessary routers
    app.include_router(home.router)
    app.include_router(data.router)
    app.include_router(dashboard.router)

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
        asyncio.create_task(background_scheduler()) # For metrics
        asyncio.create_task(model_updater_scheduler()) # For model hot-swapping

    return app

# Create the app instance
app = create_app()
