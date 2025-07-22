# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skops import io as skops_io

# Setup directories
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
MODEL_DIR = os.path.join(ROOT_DIR, 'Actions', 'Model')
RESULTS_DIR = os.path.join(ROOT_DIR, 'Actions', 'Results')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def preprocess_for_vectorizer(text):
    """Simple preprocessing function for the Vectorizer"""
    # Standardize text
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"http", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)
    text = text.replace("@", " at ")
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_and_preprocess_data():
    """Load the raw data"""
    print("Loading data...")
    
    raw_path = os.path.join(DATA_DIR, 'ci_train_data.csv')
    df = pd.read_csv(raw_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df['Review'], df['Sentiment'], test_size=0.2, random_state=42
    )
    
    print(f"Data loaded:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the TF-IDF + Logistic Regression pipeline"""
    print("Training model...")
    
    # Create pipeline with vectorizer and model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=preprocess_for_vectorizer,
            stop_words='english',
            ngram_range=(1, 3), 
            max_features=10000,
            token_pattern=r'\b[A-Za-z][A-Za-z]+\b'
        )),
        ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42))
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Save pipeline as single model
    skops_io.dump(pipeline, os.path.join(MODEL_DIR, 'sentiment_pipeline.skops'))
    
    print(f"Pipeline saved to {MODEL_DIR}")
    
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the trained pipeline"""
    print("Evaluating model...")
    
    # Make predictions using the pipeline
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(report)
    
    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(report)
    print(f"Metrics saved to {metrics_path}")
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")
    
    return accuracy

def main():
    """Main training pipeline"""
    print("Starting TripAdvisor Sentiment Analysis Training Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        
        # Step 2: Train model
        pipeline = train_model(X_train, y_train)
        
        # Step 3: Evaluate model
        accuracy = evaluate_model(pipeline, X_test, y_test)
        
        print("=" * 60)
        print(f"Training pipeline completed successfully!")
        print(f"Final accuracy: {accuracy * 100:.2f}%")
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
