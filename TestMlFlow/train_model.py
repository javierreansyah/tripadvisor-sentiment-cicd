import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import os

def load_and_preprocess_data(file_path):
    """Load and preprocess the sentiment analysis data"""
    df = pd.read_csv(file_path)
    
    # Basic text cleaning
    df['Review'] = df['Review'].astype(str)
    df['Review'] = df['Review'].str.lower()
    df['Review'] = df['Review'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['Review'] = df['Review'].str.replace(r'\s+', ' ', regex=True)
    df['Review'] = df['Review'].str.strip()
    
    return df['Review'], df['Sentiment']

def create_pipeline():
    """Create ML pipeline with TF-IDF and Logistic Regression"""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        ))
    ])
    return pipeline

def train_model():
    """Train the sentiment analysis model with MLflow tracking"""
    
    # Set MLflow experiment
    mlflow.set_experiment("Sentiment Analysis")
    
    with mlflow.start_run():
        # Load data
        print("Loading and preprocessing data...")
        X, y = load_and_preprocess_data('data.csv')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Create and train pipeline
        print("Training model...")
        pipeline = create_pipeline()
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Log parameters
        mlflow.log_param("max_features", 10000)
        mlflow.log_param("ngram_range", "(1, 2)")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Create model signature
        signature = infer_signature(X_train, y_pred_proba)
        
        # Log and register model
        model_name = "sentiment_analysis_model"
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name
        )
        
        print(f"\nModel registered as: {model_name}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return pipeline, accuracy

if __name__ == "__main__":
    # Start MLflow server in background if not running
    print("Starting MLflow tracking...")
    
    trained_model, accuracy = train_model()
    print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")
    print("\nYou can view the MLflow UI by running: mlflow ui")
