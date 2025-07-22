# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import os
import google.generativeai as genai

# Model Server URL (kept for potential future use or other integrations)
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:1234")
PREDICT_ENDPOINT = f"{MODEL_SERVER_URL}/invocations"

# MLflow Tracking URI (reads from the environment variable set in docker-compose)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

# Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
else:
    print("Warning: GEMINI_API_KEY not found. Data generation will not be available.")


# Data Directory
DATA_DIR = 'Data'

# Ensure data directory exists at startup
os.makedirs(DATA_DIR, exist_ok=True)

print(f"FastAPI app configured with MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
