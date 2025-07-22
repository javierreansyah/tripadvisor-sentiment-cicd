import asyncio
import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
import sys

from app.config import DATA_DIR
from app.metrics import (
    accuracy_gauge, 
    drift_score_gauge, 
    semantic_drift_gauge, 
    centroid_drift_gauge, 
    spread_drift_gauge
)
from .state import app_state, WINDOW_SIZE

_embedding_model = None

def get_embedding_model():
    """Get or create the embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model: all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded successfully")
    return _embedding_model

def calculate_embedding_drift_metrics(ref_embeddings, new_embeddings):
    """
    Calculate multiple drift metrics using embeddings.
    Returns individual components and combined drift score.
    """
    # Method 1: Compare centroid distances (Euclidean)
    ref_centroid = np.mean(ref_embeddings, axis=0)
    new_centroid = np.mean(new_embeddings, axis=0)
    centroid_distance = np.linalg.norm(ref_centroid - new_centroid)
    
    # Method 2: Compare embedding spreads using standard deviation
    ref_std = np.std(ref_embeddings, axis=0)
    new_std = np.std(new_embeddings, axis=0)
    spread_distance = np.linalg.norm(ref_std - new_std)
    
    # Method 3: KS test on cosine similarities to centroid
    ref_similarities = np.array([
        np.dot(emb, ref_centroid) / (np.linalg.norm(emb) * np.linalg.norm(ref_centroid))
        for emb in ref_embeddings
    ])
    new_similarities = np.array([
        np.dot(emb, ref_centroid) / (np.linalg.norm(emb) * np.linalg.norm(ref_centroid))
        for emb in new_embeddings
    ])
    semantic_ks_stat, _ = ks_2samp(ref_similarities, new_similarities)
    
    # Combine metrics into a single drift score (weighted average)
    # Normalize centroid_distance and spread_distance to [0,1] range approximately
    normalized_centroid = min(centroid_distance / 10.0, 1.0)
    normalized_spread = min(spread_distance / 5.0, 1.0)
    
    # Combined drift score: emphasize semantic similarity (KS) and structure changes
    combined_drift_score = 0.5 * semantic_ks_stat + 0.3 * normalized_centroid + 0.2 * normalized_spread
    
    return {
        'semantic_ks': semantic_ks_stat,
        'centroid_distance': centroid_distance,
        'spread_distance': spread_distance,
        'combined_drift_score': combined_drift_score
    }

async def calculate_drift_metrics():
    def _calculate_sync():
        try:
            ref_data_path = os.path.join(DATA_DIR, 'data.csv')
            new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
            if not os.path.exists(ref_data_path) or not os.path.exists(new_data_path): 
                print("Reference or new data files not found")
                return {
                    'semantic_ks': 0.0,
                    'centroid_distance': 0.0,
                    'spread_distance': 0.0,
                    'combined_drift_score': 0.0
                }
                
            df_main = pd.read_csv(ref_data_path)
            df_new = pd.read_csv(new_data_path)
            if df_new.empty: 
                print("New data is empty")
                return {
                    'semantic_ks': 0.0,
                    'centroid_distance': 0.0,
                    'spread_distance': 0.0,
                    'combined_drift_score': 0.0
                }
            
            # Get latest new data for drift calculation
            df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])
            df_new_latest = df_new.sort_values('Timestamp', ascending=False).head(WINDOW_SIZE)
            
            # Sample reference data to match size
            df_ref_sample = df_main.sample(n=len(df_new_latest), random_state=42)
            
            # Get reviews for embedding
            ref_reviews = df_ref_sample['Review'].values
            new_reviews = df_new_latest['Review'].values
            
            # Get embedding model
            embedding_model = get_embedding_model()
            
            # Generate embeddings for both datasets
            print("Generating embeddings for reference data...")
            ref_embeddings = embedding_model.encode(ref_reviews, batch_size=16, show_progress_bar=False)
            print("Generating embeddings for new data...")
            new_embeddings = embedding_model.encode(new_reviews, batch_size=16, show_progress_bar=False)
            
            # Calculate improved drift metrics using embeddings
            drift_metrics = calculate_embedding_drift_metrics(ref_embeddings, new_embeddings)
            
            print(f"Embedding-based drift metrics:")
            print(f"  Semantic KS: {drift_metrics['semantic_ks']:.4f}")
            print(f"  Centroid Distance: {drift_metrics['centroid_distance']:.4f}")
            print(f"  Spread Distance: {drift_metrics['spread_distance']:.4f}")
            print(f"  Combined Drift Score: {drift_metrics['combined_drift_score']:.4f}")
            
            return drift_metrics
            
        except Exception as e:
            print(f"Error calculating drift metrics: {e}")
            import traceback
            traceback.print_exc()
            return {
                'semantic_ks': 0.0,
                'centroid_distance': 0.0,
                'spread_distance': 0.0,
                'combined_drift_score': 0.0
            }
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
        predictions = []
        
        for review in sample_data:
            pred = model.predict([review])[0]
            predictions.append(pred)
        
        return predictions

    result = await asyncio.to_thread(_predict_sequentially_sync, X_sample)
    if not result: return 0.0
    
    predictions = result
    accuracy = accuracy_score(y_sample, predictions)
    
    return accuracy

async def calculate_and_set_all_metrics():
    """Calculate and set all metrics including accuracy and drift components."""
    accuracy, drift_metrics = await asyncio.gather(
        calculate_model_accuracy(), 
        calculate_drift_metrics()
    )
    
    # Set accuracy gauge
    accuracy_gauge.set(accuracy)
    
    # Set drift component gauges
    semantic_drift_gauge.set(drift_metrics['semantic_ks'])
    centroid_drift_gauge.set(drift_metrics['centroid_distance'])
    spread_drift_gauge.set(drift_metrics['spread_distance'])
    drift_score_gauge.set(drift_metrics['combined_drift_score'])
    
    print(f"Metrics updated:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Semantic Drift (KS): {drift_metrics['semantic_ks']:.4f}")
    print(f"  Centroid Drift: {drift_metrics['centroid_distance']:.4f}")
    print(f"  Spread Drift: {drift_metrics['spread_distance']:.4f}")
    print(f"  Combined Drift Score: {drift_metrics['combined_drift_score']:.4f}")
