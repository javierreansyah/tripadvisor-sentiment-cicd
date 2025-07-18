import os
import pandas as pd
import httpx

from app.config import DATA_DIR
from .state import app_state, WINDOW_SIZE
from .model_manager import get_latest_trained_model
from .dvc_manager import dvc_manager

# --- Orchestration Services ---

async def manage_retraining_data():
    """Appends new data to the main training set, holding back the last 200 for metrics."""
    print("Step 1: Managing and versioning data...")
    new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
    main_data_path = os.path.join(DATA_DIR, 'data.csv')
    
    # Step 1.1: Ensure data.csv is tracked by DVC
    print("Step 1.1: Ensuring data.csv is tracked by DVC...")
    if not await dvc_manager.ensure_data_tracked():
        print("Warning: Could not ensure DVC tracking for data.csv")
    
    if not os.path.exists(new_data_path): 
        print("No new data to process")
        return
    
    df_new = pd.read_csv(new_data_path)
    if df_new.empty: 
        print("New data file is empty")
        return
    
    print(f"Processing {len(df_new)} rows from new_data.csv")
    df_new.sort_values('Timestamp', ascending=True, inplace=True)
    data_to_append = df_new.iloc[:-WINDOW_SIZE]
    
    if not data_to_append.empty:
        data_to_append.to_csv(main_data_path, mode='a', header=False, index=False)
        print(f"Appended {len(data_to_append)} rows to main training data.")
    
    data_to_keep = df_new.iloc[-WINDOW_SIZE:]
    data_to_keep.to_csv(new_data_path, index=False)
    print(f"Kept last {len(data_to_keep)} rows in new_data.csv for future monitoring.")
    
    # Step 1.2: Create DVC snapshot after data movement
    print("Step 1.2: Creating DVC snapshot of updated data...")
    rows_appended = len(data_to_append) if not data_to_append.empty else 0
    snapshot_message = f"Data update: appended {rows_appended} rows, kept {len(data_to_keep)} for monitoring"
    
    if await dvc_manager.create_data_snapshot(snapshot_message):
        print("Successfully created data snapshot with DVC")
    else:
        print("Warning: Could not create DVC snapshot")

async def trigger_training_run():
    """Calls the training-runner service to start a new MLflow run."""
    print("Step 2: Triggering training-runner service...")
    training_url = "http://training-runner:5002/start-training"
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(training_url)
            response.raise_for_status()
        print("Successfully triggered training run.")
        return True
    except httpx.RequestError as e:
        print(f"Error calling training-runner service: {e}")
        return False

async def run_retraining_pipeline():
    """The main orchestration function for the background task."""
    app_state["retraining"]["is_training"] = True
    app_state["retraining"]["message"] = "Step 1/3: Preparing and versioning data..."
    await manage_retraining_data()
    app_state["retraining"]["message"] = "Step 2/3: Training new model..."
    if await trigger_training_run():
        app_state["retraining"]["message"] = "Step 3/3: Loading new model..."
        await get_latest_trained_model()
    app_state["retraining"]["is_training"] = False
    app_state["retraining"]["message"] = "Idle"
    print("Retraining pipeline finished.")
