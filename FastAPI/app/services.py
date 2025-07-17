import asyncio
import os
import pandas as pd
import aiohttp
import mlflow
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from app.config import DATA_DIR
from app.metrics import accuracy_gauge, KS_gauge, Wasserstein_gauge

# --- Model Cache ---
# A simple dictionary to hold our model in memory
model_cache = {"model": None, "version": None}
MODEL_NAME = "sentiment-classifier"
MODEL_ALIAS = "prod"

# --- Model Loading Logic ---
async def load_and_cache_model():
    """
    Loads the latest model from MLflow with the 'prod' alias.
    This function runs in a separate thread to avoid blocking the event loop.
    """
    print(f"Attempting to load model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'...")
    try:
        client = mlflow.tracking.MlflowClient()
        latest_version_info = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
        
        if model_cache["version"] == latest_version_info.version:
            print(f"Model version {latest_version_info.version} is already loaded.")
            return

        print(f"New model version found: {latest_version_info.version}. Loading...")
        model = await asyncio.to_thread(mlflow.pyfunc.load_model, model_uri=latest_version_info.source)
        
        model_cache["model"] = model
        model_cache["version"] = latest_version_info.version
        print(f"Successfully loaded and cached model version: {model_cache['version']}")

    except Exception as e:
        print(f"Error loading model: {e}")
        if model_cache["model"] is None:
            raise RuntimeError("Failed to load model on initial startup.")

# --- Monitoring Metrics Logic (Existing Code) ---
async def calculate_drift_metrics():
    """
    Asynchronously calculates data drift metrics (KS-test, Wasserstein distance).
    """
    def _calculate_sync():
        try:
            ref_data_path = os.path.join(DATA_DIR, 'test.csv')
            new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
            
            df_ref = pd.read_csv(ref_data_path)
            df_new = pd.read_csv(new_data_path)

            if df_new.empty:
                return 0.0, 0.0

            df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])
            df_new_latest = df_new.sort_values('Timestamp', ascending=False).head(100)

            ref_values = df_ref['Sentiment'].values
            new_values = df_new_latest['Sentiment'].values

            ks_stat, _ = ks_2samp(ref_values, new_values)
            wass_dist = wasserstein_distance(ref_values, new_values)
            return ks_stat, wass_dist
        except Exception as e:
            print(f"Error calculating drift metrics: {e}")
            return 0.0, 0.0

    return await asyncio.to_thread(_calculate_sync)


async def calculate_model_accuracy():
    """
    Asynchronously calculates model accuracy using the in-memory model,
    predicting one sample at a time.
    """
    def _prepare_sample_sync():
        raw_data_path = os.path.join(DATA_DIR, 'data.csv')
        if not os.path.exists(raw_data_path):
            return None, None
        df = pd.read_csv(raw_data_path)
        _, X_test, _, y_test = train_test_split(
            df['Review'], df['Sentiment'], test_size=0.2, random_state=42
        )
        sample_size = 20
        X_sample = X_test.sample(n=min(sample_size, len(X_test)))
        y_sample = y_test.loc[X_sample.index]
        return X_sample, y_sample

    X_sample, y_sample = await asyncio.to_thread(_prepare_sample_sync)

    if X_sample is None or model_cache["model"] is None:
        print("Accuracy check: No sample data or model available.")
        return 0.0

    # This function will be run in a thread to perform the CPU-bound prediction loop
    def _predict_sequentially_sync(sample_data):
        model = model_cache["model"]
        predictions_list = []
        for review in sample_data:
            # The model expects a DataFrame, even for a single item
            df_single_review = pd.DataFrame([review], columns=['Review'])
            # Predict on the single item
            prediction = model.predict(df_single_review)
            # Add the single prediction result to our list
            predictions_list.append(prediction[0])
        return predictions_list

    # Run the sequential prediction loop in a separate thread
    predictions = await asyncio.to_thread(_predict_sequentially_sync, X_sample)
        
    if not predictions:
        print("Accuracy check: Prediction loop returned no results.")
        return 0.0

    return accuracy_score(y_sample, predictions)


async def calculate_and_set_all_metrics():
    """Calculates all metrics and updates the Prometheus gauges."""
    # Run calculations sequentially to ensure model is used correctly
    drift_metrics = asyncio.create_task(calculate_drift_metrics())
    accuracy = await calculate_model_accuracy()
    ks_stat, wd_dist = await drift_metrics
    
    accuracy_gauge.set(accuracy)
    KS_gauge.set(ks_stat)
    Wasserstein_gauge.set(wd_dist)
    print(f"Metrics updated - Accuracy: {accuracy:.4f}, KS: {ks_stat:.4f}, WD: {wd_dist:.4f}")
