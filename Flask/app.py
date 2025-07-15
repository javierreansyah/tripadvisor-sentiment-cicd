import os
import time
import threading
import pandas as pd
from flask import Flask
from skops import io as skops_io
from sklearn.metrics import accuracy_score
import nltk
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge

# --- SETUP ---
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# The gauge is a global object that the background thread will update.
accuracy_gauge = Gauge('model_accuracy', 'Current accuracy of the prediction model')

# --- METRIC CALCULATION LOGIC ---
def update_accuracy_metric():
    """
    Loads data, calculates model accuracy, and updates the Prometheus gauge.
    """
    try:
        print("BACKGROUND: Starting accuracy calculation...")
        # Define paths relative to the app.py location
        MODEL_DIR = 'Model'
        DATA_DIR = 'Data'

        model_path = os.path.join(MODEL_DIR, 'logreg_tfidf.skops')
        vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.skops')

        # Load model and vectorizer
        model = skops_io.load(model_path)
        vectorizer = skops_io.load(vectorizer_path, trusted=[nltk.tokenize.word_tokenize])

        # Load test data
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        X_test = test_df['Review']
        y_test = test_df['Sentiment']

        # Transform test data and predict
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate accuracy and update the gauge
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_gauge.set(accuracy)

        print(f"BACKGROUND: Model accuracy gauge set to: {accuracy}")

    except Exception as e:
        print(f"BACKGROUND: Error updating accuracy metric: {e}")

# --- BACKGROUND SCHEDULER ---
def background_scheduler():
    """
    Runs the metric update function in a loop every 60 seconds.
    """
    while True:
        update_accuracy_metric()
        time.sleep(60)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return "Flask metrics exporter is running with a background scheduler."

# --- START BACKGROUND THREAD ---
# Run the initial calculation immediately
update_accuracy_metric()

# Create and start the background thread
# daemon=True ensures the thread will exit when the main application exits
scheduler_thread = threading.Thread(target=background_scheduler, daemon=True)
scheduler_thread.start()