import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient

def load_model_from_registry(model_name, stage="latest"):
    """Load model from MLflow model registry"""
    client = MlflowClient()
    
    try:
        if stage == "latest":
            # Get the latest version
            latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            if not latest_versions:
                raise Exception(f"No versions found for model {model_name}")
            
            # Sort by version number and get the latest
            latest_version = max(latest_versions, key=lambda x: int(x.version))
            model_version = latest_version.version
            model_uri = f"models:/{model_name}/{model_version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"
        
        print(f"Loading model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        return model, model_uri
        
    except Exception as e:
        print(f"Error loading model from registry: {e}")
        print("Trying to load from latest run...")
        
        # Fallback: load from latest run
        experiment = mlflow.get_experiment_by_name("Sentiment Analysis")
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs.empty:
                latest_run_id = runs.iloc[0]['run_id']
                model_uri = f"runs:/{latest_run_id}/model"
                print(f"Loading model from run: {model_uri}")
                model = mlflow.sklearn.load_model(model_uri)
                return model, model_uri
        
        raise Exception("Could not load model from registry or runs")

def preprocess_text(text):
    """Preprocess text in the same way as training data"""
    if isinstance(text, str):
        text = text.lower()
        text = pd.Series([text]).str.replace(r'[^\w\s]', ' ', regex=True).iloc[0]
        text = pd.Series([text]).str.replace(r'\s+', ' ', regex=True).iloc[0]
        text = text.strip()
    return text

def predict_sentiment(model, texts):
    """Make predictions on new texts"""
    if isinstance(texts, str):
        texts = [texts]
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Make predictions
    predictions = model.predict(processed_texts)
    probabilities = model.predict_proba(processed_texts)
    
    results = []
    for i, text in enumerate(texts):
        sentiment = "Positive" if predictions[i] == 1 else "Negative"
        confidence = max(probabilities[i])
        
        results.append({
            'text': text[:100] + "..." if len(text) > 100 else text,
            'sentiment': sentiment,
            'confidence': f"{confidence:.4f}",
            'positive_prob': f"{probabilities[i][1]:.4f}",
            'negative_prob': f"{probabilities[i][0]:.4f}"
        })
    
    return results

def load_test_data(file_path, n_samples=5):
    """Load some test samples from the dataset"""
    df = pd.read_csv(file_path)
    sample_df = df.sample(n=n_samples, random_state=42)
    return sample_df

def main():
    print("=== MLflow Sentiment Analysis Prediction ===\n")
    
    model_name = "sentiment_analysis_model"
    
    try:
        # Load model from registry
        print("Loading model from MLflow registry...")
        model, model_uri = load_model_from_registry(model_name)
        print(f"Model loaded successfully from: {model_uri}\n")
        
        # Option 1: Predict on sample data from CSV
        print("1. Testing on sample data from CSV:")
        print("-" * 50)
        test_samples = load_test_data('data.csv', n_samples=3)
        
        for idx, row in test_samples.iterrows():
            actual_sentiment = "Positive" if row['Sentiment'] == 1 else "Negative"
            
            results = predict_sentiment(model, [row['Review']])
            result = results[0]
            
            print(f"Review: {result['text']}")
            print(f"Actual Sentiment: {actual_sentiment}")
            print(f"Predicted Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Positive Probability: {result['positive_prob']}")
            print(f"Negative Probability: {result['negative_prob']}")
            print("-" * 50)
        
        # Option 2: Predict on custom text examples
        print("\n2. Testing on custom examples:")
        print("-" * 50)
        
        custom_texts = [
            "This hotel was absolutely amazing! Great service and beautiful rooms.",
            "Terrible experience. The room was dirty and the staff was rude.",
            "The hotel was okay, nothing special but clean and comfortable.",
            "I loved staying here! Will definitely come back again.",
            "Worst hotel ever. Would not recommend to anyone."
        ]
        
        for text in custom_texts:
            results = predict_sentiment(model, [text])
            result = results[0]
            
            print(f"Text: {result['text']}")
            print(f"Predicted Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']}")
            print("-" * 30)
        
        # Option 3: Interactive prediction
        print("\n3. Interactive prediction (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            user_input = input("\nEnter a hotel review to analyze: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                results = predict_sentiment(model, [user_input])
                result = results[0]
                
                print(f"Predicted Sentiment: {result['sentiment']}")
                print(f"Confidence: {result['confidence']}")
                print(f"Positive Probability: {result['positive_prob']}")
                print(f"Negative Probability: {result['negative_prob']}")
            else:
                print("Please enter some text to analyze.")
        
        print("\nPrediction session ended.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Trained the model by running train_model.py")
        print("2. Started MLflow server (mlflow ui)")
        print("3. The model is registered in MLflow")

if __name__ == "__main__":
    main()
