# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import os
import pandas as pd
import re
import mlflow
import mlflow.sklearn
import joblib
import tempfile
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Sentiment Analysis")

mlflow.sklearn.autolog(
    log_model_signatures=False,
    log_models=False,
    log_datasets=False,
    log_input_examples=False,
    log_post_training_metrics=False,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False
)

print(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
print("MLflow sklearn autologging enabled")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
os.makedirs(DATA_DIR, exist_ok=True)

# Preprocessing
def preprocess_for_vectorizer(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"http", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)
    text = text.replace("@", " at ")
    text = text.lower()
    text = ' '.join(text.split())
    return text


# Training and Evaluation Logic
def run_training_pipeline():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get existing runs count for versioning
    experiment = mlflow.get_experiment_by_name("Sentiment Analysis")
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        version_number = len(runs) + 1
    else:
        version_number = 1
    
    run_name = f"sentiment_model_v{version_number}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Starting MLflow Run: {run.info.run_id}")
        print(f"Run Name: {run_name}")
        
        # Add tags for better organization
        mlflow.set_tag("model_type", "sentiment_classifier")
        mlflow.set_tag("version", f"v{version_number}")
        mlflow.set_tag("algorithm", "logistic_regression")
        mlflow.set_tag("feature_extraction", "tfidf")
        mlflow.set_tag("created_by", "training_pipeline")

        print("--- Step 1: Loading Raw Data ---")
        raw_path = os.path.join(DATA_DIR, 'data.csv')
        df = pd.read_csv(raw_path)
        
        X = df['Review']
        y = df['Sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Raw data split complete.")

        print("--- Step 2: Training Model ---")
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                preprocessor=preprocess_for_vectorizer,
                stop_words='english',
                ngram_range=(1, 3), 
                max_features=10000,
                token_pattern=r'\b[A-Za-z][A-Za-z]+\b'
            )),
            ('classifier', LogisticRegression(
                max_iter=1000, 
                solver="lbfgs",
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)
        print("Training complete.")

        print("--- Step 3: Evaluating Model ---")
        # Calculate predictions and probabilities
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability for positive class
        
        # Calculate all test metrics to match training metrics
        test_accuracy_score = accuracy_score(y_test, y_pred)
        test_precision_score = precision_score(y_test, y_pred, average='macro')
        test_recall_score = recall_score(y_test, y_pred, average='macro')
        test_f1_score = f1_score(y_test, y_pred, average='macro')
        test_roc_auc = roc_auc_score(y_test, y_proba)
        test_log_loss = log_loss(y_test, y_proba)
        test_score = test_accuracy_score  # Same as accuracy for consistency
        
        # Log all test metrics with same names as training (but test_ prefix)
        mlflow.log_metric("test_accuracy_score", test_accuracy_score)
        mlflow.log_metric("test_precision_score", test_precision_score)
        mlflow.log_metric("test_recall_score", test_recall_score)
        mlflow.log_metric("test_f1_score", test_f1_score)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("test_log_loss", test_log_loss)
        mlflow.log_metric("test_score", test_score)
        
        print(f"Test Accuracy: {test_accuracy_score * 100:.2f}%")
        print(f"Test Metrics - Precision: {test_precision_score:.4f}, Recall: {test_recall_score:.4f}, F1: {test_f1_score:.4f}, ROC-AUC: {test_roc_auc:.4f}")

        # LOG CONFUSION MATRIX AS ARTIFACT
        print("--- Step 4: Logging Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Create properly named file
        temp_dir = tempfile.gettempdir()
        confusion_matrix_path = os.path.join(temp_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        
        # Log to evaluation_plots directory
        mlflow.log_artifact(confusion_matrix_path, "evaluation_plots")
        print(f"Confusion matrix saved and logged.")
        
        # Clean up
        os.unlink(confusion_matrix_path)

        # LOG ROC CURVE AS ARTIFACT
        print("--- Step 4b: Logging ROC Curve ---")
        # Get prediction probabilities for positive class (already calculated above)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create properly named file
        roc_curve_path = os.path.join(temp_dir, 'roc_curve.png')
        plt.savefig(roc_curve_path)
        plt.close()
        
        # Log to evaluation_plots directory
        mlflow.log_artifact(roc_curve_path, "evaluation_plots")
        print(f"ROC curve saved and logged (AUC = {test_roc_auc:.3f}).")
        
        # Clean up
        os.unlink(roc_curve_path)

        # LOG CLASSIFICATION REPORT AS TXT
        print("--- Step 4c: Logging Classification Report ---")
        report_text = classification_report(y_test, y_pred)
        
        # Create properly named file
        classification_report_path = os.path.join(temp_dir, 'classification_report.txt')
        with open(classification_report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("====================\n\n")
            f.write(report_text)
            f.write(f"\n\nROC AUC Score: {test_roc_auc:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy_score:.4f}\n")
            f.write(f"Test Precision (Macro): {test_precision_score:.4f}\n")
            f.write(f"Test Recall (Macro): {test_recall_score:.4f}\n")
            f.write(f"Test F1 Score (Macro): {test_f1_score:.4f}\n")
            f.write(f"Test Log Loss: {test_log_loss:.4f}\n")
            
        # Log to evaluation_reports directory
        mlflow.log_artifact(classification_report_path, "evaluation_reports")
        print("Classification report saved and logged.")
        
        # Clean up
        os.unlink(classification_report_path)

        # MANUAL DATASET LOGGING
        print("--- Step 4d: Manual Dataset Logging ---")
        
        # Log dataset information to MLflow UI (not as artifact)
        dataset = mlflow.data.from_pandas(df, source=raw_path, name="sentiment_dataset", targets="Sentiment")
        mlflow.log_input(dataset, context="training")
        print("Dataset information logged to MLflow UI.")
        
        # Create comprehensive data summary
        summary = {
            "dataset_info": {
                "original_dataset_path": "data.csv",
                "total_samples": len(df),
                "train_samples": len(X_train), 
                "test_samples": len(X_test),
                "test_split_ratio": 0.2,
                "features": ["Review"],
                "target": "Sentiment"
            },
            "class_distribution": {
                "overall": {
                    "negative (0)": int((y == 0).sum()),
                    "positive (1)": int((y == 1).sum())
                },
                "train": {
                    "negative (0)": int((y_train == 0).sum()),
                    "positive (1)": int((y_train == 1).sum())
                },
                "test": {
                    "negative (0)": int((y_test == 0).sum()),
                    "positive (1)": int((y_test == 1).sum())
                }
            },
            "text_statistics": {
                "avg_review_length_chars": int(X.str.len().mean()),
                "max_review_length_chars": int(X.str.len().max()),
                "min_review_length_chars": int(X.str.len().min()),
                "avg_review_length_words": int(X.str.split().str.len().mean()),
                "train_avg_length_chars": int(X_train.str.len().mean()),
                "test_avg_length_chars": int(X_test.str.len().mean())
            },
            "model_performance": {
                "test_accuracy": float(test_accuracy_score),
                "test_precision": float(test_precision_score),
                "test_recall": float(test_recall_score),
                "test_f1_score": float(test_f1_score),
                "test_roc_auc": float(test_roc_auc),
                "test_log_loss": float(test_log_loss)
            }
        }
        
        # Create named file for data summary
        data_summary_path = os.path.join(temp_dir, 'data_summary.json')
        with open(data_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Log to data_summary directory
        mlflow.log_artifact(data_summary_path, "data_summary")
        print("Data summary saved and logged.")
        
        # Clean up
        os.unlink(data_summary_path)

        # LOG AND REGISTER THE MODEL (manual approach for reliability)
        print("--- Step 5: Logging and Registering Model ---")
        
        # Log the model manually (more reliable than autologging)
        signature = mlflow.models.infer_signature(X_test, pipeline.predict(X_test))
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="sentiment-model",
            signature=signature,
            input_example=X_test.iloc[:5].tolist(),
        )
        print("Model logged successfully.")
        
        # Register the model only if test accuracy is above 80%
        if test_accuracy_score > 0.8:
            model_uri = f"runs:/{run.info.run_id}/sentiment-model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name="sentiment-classifier"
            )
            print(f"Model registered with version: {registered_model.version}")
            print(f"Model passed validation with test accuracy: {test_accuracy_score * 100:.2f}%")
        else:
            print(f"Model NOT registered - test accuracy {test_accuracy_score * 100:.2f}% is below 80% threshold")
            print("Model logged but not registered due to low performance")

        print("Pipeline successfully completed.")
        print("Autologging captured: hyperparameters and training metrics automatically.")
        print("Model logged manually for reliability.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        
        # Final artifact check
        try:
            client = mlflow.tracking.MlflowClient()
            final_artifacts = client.list_artifacts(run.info.run_id)
            print("Final artifacts in this run:")
            for artifact in final_artifacts:
                print(f"  - {artifact.path}")
        except Exception as e:
            print(f"Could not list final artifacts: {e}")

if __name__ == "__main__":
    run_training_pipeline()