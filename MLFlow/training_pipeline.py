import os
import pandas as pd
import re
import nltk
import mlflow
import mlflow.sklearn
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Sentiment Analysis")

print(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

# --- Directory Setup ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'Results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Preprocessing function for the Vectorizer ---
# This function will be used *inside* the pipeline's vectorizer
def preprocess_for_vectorizer(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Standardize text
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"http", "", text)
    text = re.sub(r"@/S+", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)
    text = text.replace("@", " at ")
    text = text.lower()
    
    # Tokenize, remove stopwords, and lemmatize
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


# --- Main Training and Evaluation Logic ---
def run_training_pipeline():
    with mlflow.start_run() as run:
        print(f"Starting MLflow Run: {run.info.run_id}")

        # 1. LOAD RAW DATA
        print("--- Step 1: Loading Raw Data ---")
        raw_path = os.path.join(DATA_DIR, 'data.csv')
        df = pd.read_csv(raw_path)
        
        # Split the RAW data. The pipeline will handle preprocessing.
        X = df['Review']
        y = df['Sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Raw data split complete.")

        # 2. TRAINING
        print("--- Step 2: Training Model ---")
        params = {
            "ngram_range": (1, 3),
            "max_features": 10000,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42
        }
        mlflow.log_params(params)

        # Create a scikit-learn pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                preprocessor=preprocess_for_vectorizer,
                tokenizer=word_tokenize,
                ngram_range=params["ngram_range"], 
                max_features=params["max_features"]
            )),
            ('classifier', LogisticRegression(
                max_iter=params["max_iter"], 
                solver=params["solver"],
                random_state=params["random_state"]
            ))
        ])

        # Fit the entire pipeline on the RAW training data
        pipeline.fit(X_train, y_train)
        print("Training complete.")

        # 3. EVALUATION
        print("--- Step 3: Evaluating Model ---")
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("precision_weighted", report_dict["weighted avg"]["precision"])
        mlflow.log_metric("recall_weighted", report_dict["weighted avg"]["recall"])
        mlflow.log_metric("f1_score_weighted", report_dict["weighted avg"]["f1-score"])

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        plot_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path, "evaluation_plots")
        print(f"Confusion matrix saved and logged.")

        # 4. LOGGING THE MODEL
        print("--- Step 4: Logging Model to MLflow ---")
        signature = mlflow.models.infer_signature(X_test, pipeline.predict(X_test))
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="sentiment-model",
            signature=signature,
            input_example=X_test.iloc[:5].tolist(),
        )
        
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/sentiment-model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="sentiment-classifier"
        )
        print(f"Model registered with version: {registered_model.version}")
        
        # 5. EXPORT VECTORIZER AS SEPARATE ARTIFACT
        print("--- Step 5: Logging Vectorizer Artifact ---")
        # Extract the vectorizer from the pipeline
        vectorizer = pipeline.named_steps['vectorizer']
        
        # Save vectorizer using joblib for better compatibility
        vectorizer_path = os.path.join(RESULTS_DIR, 'vectorizer.pkl')
        joblib.dump(vectorizer, vectorizer_path)
        
        # Log as artifact
        mlflow.log_artifact(vectorizer_path, "preprocessing")
        print("Vectorizer saved and logged as artifact.")
        
        print("Pipeline successfully logged and registered to MLflow.")

if __name__ == "__main__":
    run_training_pipeline()