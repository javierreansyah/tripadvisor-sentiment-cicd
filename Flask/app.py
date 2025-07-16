import os
import time
import threading
import pandas as pd
from flask import Flask
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge

# --- SETUP ---
app = Flask(__name__)
metrics = PrometheusMetrics(app)
accuracy_gauge = Gauge('model_accuracy', 'Current accuracy of the prediction model')

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:1234")
PREDICT_ENDPOINT = f"{MODEL_SERVER_URL}/invocations"

print(f"Flask app configured to make predictions to: {PREDICT_ENDPOINT}")

# --- METRIC CALCULATION LOGIC ---
def update_accuracy_metric():
    """
    Loads data, sends each review separately to the model server for prediction,
    and updates the Prometheus gauge with the sample's accuracy.
    """
    try:
        print("BACKGROUND: Starting accuracy calculation with a random sample...")
        DATA_DIR = 'Data'

        raw_data_path = os.path.join(DATA_DIR, 'data.csv')
        df = pd.read_csv(raw_data_path)
        _, X_test, _, y_test = train_test_split(
            df['Review'], df['Sentiment'], test_size=0.2, random_state=42
        )

        sample_size = 20
        X_sample = X_test.sample(n=sample_size)
        y_sample = y_test.loc[X_sample.index]
        
        print(f"BACKGROUND: Created a random sample of {len(X_sample)} reviews.")

        predictions = []
        for idx, review in enumerate(X_sample):
            json_data = {
                "dataframe_split": pd.DataFrame([review], columns=['Review']).to_dict(orient="split")
            }
            
            print(f"BACKGROUND: Sending request {idx+1}/{sample_size} to model server...")
            response = requests.post(PREDICT_ENDPOINT, json=json_data, timeout=10)

            if response.status_code == 200:
                response_json = response.json()
                pred = response_json.get('predictions')
                if pred is not None and len(pred) > 0:
                    predictions.append(pred[0])
                    print(f"BACKGROUND: Received prediction: {pred[0]}")
                else:
                    print(f"BACKGROUND: Warning - 'predictions' missing or empty in response. Content: {response.text}")
                    predictions.append(None)  # mark as None if failed
            else:
                print(f"BACKGROUND: Error - Non-200 status code: {response.status_code}. Content: {response.text}")
                predictions.append(None)

        # Filter out failed predictions before accuracy calculation
        valid_indices = [i for i, p in enumerate(predictions) if p is not None]
        y_valid = y_sample.iloc[valid_indices]
        predictions_valid = [predictions[i] for i in valid_indices]

        if len(predictions_valid) > 0:
            accuracy = accuracy_score(y_valid, predictions_valid)
            accuracy_gauge.set(accuracy)
            print(f"BACKGROUND: Sample accuracy gauge set to: {accuracy}")
        else:
            print("BACKGROUND: All predictions failed; accuracy cannot be computed.")

    except requests.exceptions.RequestException as e:
        print(f"BACKGROUND: Network error connecting to model server: {e}")
    except Exception as e:
        print(f"BACKGROUND: An unexpected error occurred: {e}")

# --- BACKGROUND SCHEDULER ---
def background_scheduler():
    while True:
        update_accuracy_metric()
        time.sleep(60)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return f"Flask metrics exporter is running. Using model server at {MODEL_SERVER_URL}"

# --- START BACKGROUND THREAD ---
scheduler_thread = threading.Thread(target=background_scheduler, daemon=True)
scheduler_thread.start()

# --- RUN FLASK ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
