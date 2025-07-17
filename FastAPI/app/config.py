import os

# --- APPLICATION CONFIGURATION ---

# Model Server URL (kept for potential future use or other integrations)
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:1234")
PREDICT_ENDPOINT = f"{MODEL_SERVER_URL}/invocations"

# MLflow Tracking URI (reads from the environment variable set in docker-compose)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

# Data Directory
DATA_DIR = 'Data'

# Ensure data directory exists at startup
os.makedirs(DATA_DIR, exist_ok=True)

print(f"FastAPI app configured with MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
