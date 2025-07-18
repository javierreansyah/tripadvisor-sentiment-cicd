import asyncio
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

from .state import app_state, MODEL_NAME, MODEL_ALIAS

# --- Model Loading & State Update Service ---

async def load_and_cache_model():
    """Loads the latest production model and updates the central app_state."""
    print(f"State Update: Checking for production model '{MODEL_NAME}@{MODEL_ALIAS}'...")
    client = MlflowClient()
    try:
        latest_version_info = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
        
        if app_state["model_cache"]["version"] == latest_version_info.version:
            return

        print(f"State Update: New model version {latest_version_info.version} found. Loading...")
        model = await asyncio.to_thread(mlflow.pyfunc.load_model, model_uri=latest_version_info.source)
        
        run_id = latest_version_info.run_id
        
        app_state["model_cache"]["model"] = model
        app_state["model_cache"]["version"] = latest_version_info.version
        app_state["model_cache"]["model_info"] = {
            "run_id": run_id,
            "version": latest_version_info.version
        }
        accuracy = 0.0
        if run_id:
            try:
                run_info = client.get_run(run_id)
                accuracy = run_info.data.metrics.get("accuracy", 0.0)
            except Exception as run_e:
                print(f"Warning: Could not fetch run details for run_id '{run_id}'. Error: {run_e}")
        else:
            print(f"Warning: Model version {latest_version_info.version} is not linked to a specific run.")

        app_state["champion"] = {
            "version": latest_version_info.version,
            "run_id": run_id if run_id else "N/A",
            "accuracy": accuracy
        }
        print(f"State Update: Successfully loaded and cached model version {app_state['model_cache']['version']}.")

    except RestException:
        print("State Update: No model found with '@prod' alias.")
        app_state["champion"] = None
        app_state["model_cache"] = {"model": None, "version": None, "model_info": None}
    except Exception as e:
        print(f"State Update: An unexpected error occurred while loading the model: {e}")
        app_state["champion"] = None
        if app_state["model_cache"]["model"] is None:
            raise RuntimeError("Failed to load model on initial startup.")

async def get_latest_trained_model():
    """Gets information about the latest trained model."""
    print("Getting latest trained model information...")
    client = MlflowClient()
    
    try:
        # Get the latest version
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            print("No model versions found.")
            return
            
        latest_version = versions[0]  # Already sorted by version number
        run_id = latest_version.run_id
        
        # Get the run information to fetch metrics
        run = client.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy", 0.0)
        
        print(f"Found latest model version: {latest_version.version}")
        app_state["retraining"]["candidate"] = {
            "version": int(latest_version.version),
            "accuracy": accuracy,
            "run_id": run_id
        }

    except Exception as e:
        print(f"An error occurred during model validation: {e}")

async def promote_model_to_prod(version: int):
    """Sets the @prod alias and triggers an immediate state update."""
    print(f"Promoting version {version} to Production...")
    client = MlflowClient()
    client.set_registered_model_alias(name=MODEL_NAME, alias="prod", version=version)
    app_state["retraining"]["candidate"] = None
    await load_and_cache_model()
    print(f"Version {version} is now in production and loaded.")
