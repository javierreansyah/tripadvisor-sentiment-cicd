from fastapi import FastAPI, HTTPException
import subprocess
import os

app = FastAPI(title="Training Runner Service")

# The path to your original training script
TRAINING_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'training_pipeline.py')

@app.get("/")
async def root():
    """
    Root endpoint with links to trigger training.
    """
    return {
        "message": "MLFlow Training Service", 
        "endpoints": {
            "start_training": "/retrain",
            "api_docs": "/docs"
        }
    }

@app.get("/retrain")
async def retrain_model():
    """
    GET endpoint to trigger retraining - can be visited directly in browser.
    """
    try:
        print("Received request to start training pipeline via GET...")
        # We run the script using the same Python executable that runs the app.
        process = subprocess.run(
            ['python', TRAINING_SCRIPT_PATH],
            capture_output=True,
            text=True,
            check=True  # This will raise an exception if the script fails
        )
        
        print("Training script executed successfully.")
        print("Script output:", process.stdout)
        
        return {
            "status": "success", 
            "message": "Training pipeline completed successfully!",
            "output": process.stdout
        }

    except subprocess.CalledProcessError as e:
        print(f"Error executing training script: {e}")
        print("Error output:", e.stderr)
        return {
            "status": "error",
            "message": "Training script failed.",
            "error_details": e.stderr
        }
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

@app.post("/start-training")
async def start_training():
    """
    POST endpoint to trigger the training pipeline script as a subprocess.
    """
    try:
        print("Received request to start training pipeline...")
        # We run the script using the same Python executable that runs the app.
        process = subprocess.run(
            ['python', TRAINING_SCRIPT_PATH],
            capture_output=True,
            text=True,
            check=True  # This will raise an exception if the script fails
        )
        
        print("Training script executed successfully.")
        print("Script output:", process.stdout)
        
        return {"status": "success", "message": "Training pipeline started."}

    except subprocess.CalledProcessError as e:
        print(f"Error executing training script: {e}")
        print("Error output:", e.stderr)
        raise HTTPException(
            status_code=500, 
            detail={"message": "Training script failed.", "details": e.stderr}
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
