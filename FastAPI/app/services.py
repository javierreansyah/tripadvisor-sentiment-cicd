import asyncio
import os
import pandas as pd
import mlflow
import csv
import pytz
import json
import google.generativeai as genai
from datetime import datetime
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import accuracy_score

from app.config import DATA_DIR
from app.metrics import accuracy_gauge, KS_gauge, Wasserstein_gauge

# --- Model Cache ---
model_cache = {"model": None, "version": None}
MODEL_NAME = "sentiment-classifier"
MODEL_ALIAS = "prod"
WINDOW_SIZE = 200

# --- Model Loading Logic ---
async def load_and_cache_model():
    """
    Loads the latest model from MLflow with the 'prod' alias.
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

# --- Monitoring Metrics Logic ---
async def calculate_drift_metrics():
    """
    Calculates data drift by comparing the latest 200 data points from
    'new_data.csv' against a random sample from the original 'data.csv'.
    """
    def _calculate_sync():
        try:
            ref_data_path = os.path.join(DATA_DIR, 'data.csv')
            new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
            
            if not os.path.exists(ref_data_path) or not os.path.exists(new_data_path):
                print("Drift check: Missing data.csv or new_data.csv")
                return 0.0, 0.0

            df_main = pd.read_csv(ref_data_path)
            df_new = pd.read_csv(new_data_path)

            if df_new.empty:
                print("Drift check: new_data.csv is empty.")
                return 0.0, 0.0

            # Get the latest data from new_data.csv based on timestamp
            df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])
            df_new_latest = df_new.sort_values('Timestamp', ascending=False).head(WINDOW_SIZE)
            new_values = df_new_latest['Sentiment'].values

            # Create a reference sample from the original data of the same size
            df_ref_sample = df_main.sample(n=len(new_values), random_state=42)
            ref_values = df_ref_sample['Sentiment'].values
            
            ks_stat, _ = ks_2samp(ref_values, new_values)
            wass_dist = wasserstein_distance(ref_values, new_values)
            return ks_stat, wass_dist
            
        except Exception as e:
            print(f"Error calculating drift metrics: {e}")
            return 0.0, 0.0

    return await asyncio.to_thread(_calculate_sync)

async def calculate_model_accuracy():
    """
    Calculates model accuracy on the latest 200 data points from 'new_data.csv'.
    """
    def _get_latest_data_for_accuracy():
        """Fetches the latest data window from new_data.csv."""
        new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
        if not os.path.exists(new_data_path):
            return None, None

        df_new = pd.read_csv(new_data_path)
        if df_new.empty:
            return None, None
        
        df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])
        df_new_latest = df_new.sort_values('Timestamp', ascending=False).head(WINDOW_SIZE)
        
        X_sample = df_new_latest['Review']
        y_sample = df_new_latest['Sentiment']
        return X_sample, y_sample

    X_sample, y_sample = await asyncio.to_thread(_get_latest_data_for_accuracy)

    if X_sample is None or y_sample is None or model_cache["model"] is None:
        print("Accuracy check: No recent data or model available.")
        return 0.0

    def _predict_sequentially_sync(sample_data):
        model = model_cache["model"]
        predictions_list = []
        for review in sample_data:
            df_single_review = pd.DataFrame([review], columns=['Review'])
            prediction = model.predict(df_single_review)
            predictions_list.append(prediction[0])
        return predictions_list

    predictions = await asyncio.to_thread(_predict_sequentially_sync, X_sample)
        
    if not predictions:
        print("Accuracy check: Prediction loop returned no results.")
        return 0.0

    return accuracy_score(y_sample, predictions)

async def calculate_and_set_all_metrics():
    """Calculates all metrics and updates the Prometheus gauges."""
    # Run calculations concurrently as they are independent now
    accuracy, (ks_stat, wd_dist) = await asyncio.gather(
        calculate_model_accuracy(),
        calculate_drift_metrics()
    )
    
    accuracy_gauge.set(accuracy)
    KS_gauge.set(ks_stat)
    Wasserstein_gauge.set(wd_dist)
    print(f"Metrics updated - Accuracy: {accuracy:.4f}, KS: {ks_stat:.4f}, WD: {wd_dist:.4f}")

# --- Data Handling Services ---
def write_data_to_csv(data_to_write: list):
    """
    A generic function to append rows of data to the new_data.csv file.
    Each row in data_to_write should be [timestamp, review, sentiment].
    """
    new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
    header_exists = os.path.exists(new_data_path) and os.path.getsize(new_data_path) > 0
    with open(new_data_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        if not header_exists:
            writer.writerow(['Timestamp', 'Review', 'Sentiment'])
        writer.writerows(data_to_write)

def add_manual_data(review: str, sentiment: int):
    """Prepares a single manual entry and writes it to CSV."""
    wib_tz = pytz.timezone('Asia/Jakarta')
    timestamp = datetime.now(wib_tz).strftime("%Y-%m-%d %H:%M:%S")
    write_data_to_csv([[timestamp, review, sentiment]])
    print("New manual data added. Metrics will be updated on the next scheduled run.")

def generate_and_save_gemini_data(style: str, quantity: int):
    """Generates data from Gemini and writes it to CSV."""
    prompt = f"""
    Generate a list of {quantity} realistic hotel reviews.
    The style of the reviews must be: {style}.
    For each review, provide a sentiment score: 1 for a positive review and 0 for a negative review.
    Ensure each review is concise and under 80 words.
    Include a mix of both positive and negative experiences.
    """
    try:
        review_schema = {
            "type": "ARRAY", "items": {
                "type": "OBJECT", "properties": {
                    "Review": {"type": "STRING"}, "Sentiment": {"type": "INTEGER"}
                }, "required": ["Review", "Sentiment"]
            }
        }
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json", response_schema=review_schema
        )
        model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash-latest", generation_config=generation_config
        )
        
        print(f"Generating {quantity} reviews with style '{style}'...")
        response = model.generate_content(prompt)
        reviews_data = json.loads(response.text)

        wib_tz = pytz.timezone('Asia/Jakarta')
        timestamp = datetime.now(wib_tz).strftime("%Y-%m-%d %H:%M:%S")
        data_to_write = [[timestamp, item['Review'], item['Sentiment']] for item in reviews_data]
        
        write_data_to_csv(data_to_write)
        print(f"Successfully generated and saved {len(reviews_data)} reviews. Metrics will be updated soon.")
        return len(reviews_data)
    except Exception as e:
        print(f"An error occurred during data generation: {e}")
        # Re-raise the exception to be handled by the router
        raise e
