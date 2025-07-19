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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skops import io as skops_io

# Setup directories
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
MODEL_DIR = os.path.join(ROOT_DIR, 'Model')
RESULTS_DIR = os.path.join(ROOT_DIR, 'Results')

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
    
    # Load raw data
    raw_path = os.path.join(DATA_DIR, 'ci_train_data.csv')
    df = pd.read_csv(raw_path)
    
    # Split data without preprocessing (vectorizer will handle it)
    X_train, X_test, y_train, y_test = train_test_split(
        df['Review'], df['Sentiment'], test_size=0.2, random_state=42
    )
    
    print(f"Data loaded:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the TF-IDF + Logistic Regression model"""
    print("Training model...")
    
    # Create and fit vectorizer with built-in preprocessing
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_for_vectorizer,
        stop_words='english',  # Use sklearn's built-in English stopwords
        ngram_range=(1, 3), 
        max_features=10000,
        token_pattern=r'\b[A-Za-z][A-Za-z]+\b'  # Only alphabetic tokens with 2+ characters
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Save model and vectorizer
    skops_io.dump(model, os.path.join(MODEL_DIR, 'logreg_tfidf.skops'))
    skops_io.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.skops'))
    
    print(f"Model and vectorizer saved to {MODEL_DIR}")
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate the trained model"""
    print("Evaluating model...")
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    
    print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
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
        model, vectorizer = train_model(X_train, y_train)
        
        # Step 3: Evaluate model
        accuracy = evaluate_model(model, vectorizer, X_test, y_test)
        
        print("=" * 60)
        print(f"🎉 Training pipeline completed successfully!")
        print(f"Final accuracy: {accuracy * 100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
