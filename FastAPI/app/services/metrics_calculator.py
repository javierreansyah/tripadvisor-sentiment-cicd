import asyncio
import os
import pandas as pd
import mlflow
import joblib
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import accuracy_score
import sys

from app.config import DATA_DIR
from app.metrics import accuracy_gauge, KS_gauge, Wasserstein_gauge
from app.preprocessing import preprocess_for_vectorizer
from .state import app_state, WINDOW_SIZE

async def calculate_drift_metrics():
    def _calculate_sync():
        try:
            # Get the latest trained model info to retrieve the vectorizer
            model_info = app_state["model_cache"]["model_info"]
            if not model_info or "run_id" not in model_info:
                print("No model info available for vectorizer retrieval")
                return 0.0, 0.0
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri("http://mlflow:5000")
            run_id = model_info["run_id"]
            
            # Download vectorizer artifact from MLflow
            try:
                # Make the preprocessing function available in the global namespace
                # This is needed because the vectorizer was pickled with a reference to this function
                import __main__
                __main__.preprocess_for_vectorizer = preprocess_for_vectorizer
                
                print(f"Downloading vectorizer artifact for run: {run_id}")
                vectorizer_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="preprocessing/vectorizer.pkl"
                )
                print(f"Loading vectorizer from: {vectorizer_path}")
                vectorizer = joblib.load(vectorizer_path)
                print("Vectorizer loaded successfully")
            except Exception as e:
                print(f"Error loading vectorizer from MLflow: {e}")
                import traceback
                traceback.print_exc()
                return 0.0, 0.0
            
            # Load data
            ref_data_path = os.path.join(DATA_DIR, 'data.csv')
            new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
            if not os.path.exists(ref_data_path) or not os.path.exists(new_data_path): 
                return 0.0, 0.0
                
            df_main = pd.read_csv(ref_data_path)
            df_new = pd.read_csv(new_data_path)
            if df_new.empty: 
                return 0.0, 0.0
            
            # Get latest new data for drift calculation
            df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])
            df_new_latest = df_new.sort_values('Timestamp', ascending=False).head(WINDOW_SIZE)
            
            # Sample reference data to match size
            df_ref_sample = df_main.sample(n=len(df_new_latest), random_state=42)
            
            # Vectorize the review texts (not sentiments)
            ref_reviews = df_ref_sample['Review'].values
            new_reviews = df_new_latest['Review'].values
            
            # Transform using the trained vectorizer
            ref_vectors = vectorizer.transform(ref_reviews).toarray()
            new_vectors = vectorizer.transform(new_reviews).toarray()
            
            # Calculate drift metrics on vectorized features
            # For KS test, we'll use the mean of each sample's feature vector
            ref_feature_means = np.mean(ref_vectors, axis=1)
            new_feature_means = np.mean(new_vectors, axis=1)
            
            ks_stat, _ = ks_2samp(ref_feature_means, new_feature_means)
            
            # For Wasserstein distance, we'll use the mean feature vector of each dataset
            ref_centroid = np.mean(ref_vectors, axis=0)
            new_centroid = np.mean(new_vectors, axis=0)
            wass_dist = wasserstein_distance(ref_centroid, new_centroid)
            
            return ks_stat, wass_dist
            
        except Exception as e:
            print(f"Error calculating drift metrics: {e}")
            return 0.0, 0.0
    return await asyncio.to_thread(_calculate_sync)

async def calculate_model_accuracy():
    def _get_latest_data_for_accuracy():
        new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
        if not os.path.exists(new_data_path): return None, None
        df_new = pd.read_csv(new_data_path)
        if df_new.empty: return None, None
        df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])
        df_new_latest = df_new.sort_values('Timestamp', ascending=False).head(WINDOW_SIZE)
        return df_new_latest['Review'], df_new_latest['Sentiment']

    X_sample, y_sample = await asyncio.to_thread(_get_latest_data_for_accuracy)
    if X_sample is None or y_sample is None or app_state["model_cache"]["model"] is None:
        return 0.0

    def _predict_sequentially_sync(sample_data):
        model = app_state["model_cache"]["model"]
        return [model.predict(pd.DataFrame([review], columns=['Review']))[0] for review in sample_data]

    predictions = await asyncio.to_thread(_predict_sequentially_sync, X_sample)
    if not predictions: return 0.0
    return accuracy_score(y_sample, predictions)

async def calculate_and_set_all_metrics():
    accuracy, (ks_stat, wd_dist) = await asyncio.gather(calculate_model_accuracy(), calculate_drift_metrics())
    accuracy_gauge.set(accuracy)
    KS_gauge.set(ks_stat)
    Wasserstein_gauge.set(wd_dist)
    print(f"Metrics updated - Accuracy: {accuracy:.4f}, KS: {ks_stat:.4f}, WD: {wd_dist:.4f}")
