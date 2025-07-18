# --- Centralized State Management ---
app_state = {
    "retraining": {
        "is_training": False,
        "message": "Idle",
        "candidate": None
    },
    "champion": None,
    "model_cache": {"model": None, "version": None, "model_info": None}
}

# --- Constants ---
MODEL_NAME = "sentiment-classifier"
MODEL_ALIAS = "prod"
WINDOW_SIZE = 200

async def get_dashboard_data():
    """Fetches all data needed to render the dashboard from the in-memory state."""
    return {
        "status": app_state["retraining"],
        "champion": app_state["champion"],
        "candidate": app_state["retraining"]["candidate"]
    }
