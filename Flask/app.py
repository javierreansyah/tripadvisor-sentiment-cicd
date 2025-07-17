import os
import time
import threading
import pandas as pd
from flask import Flask, request, render_template_string, redirect, url_for
import requests
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge
from scipy.stats import ks_2samp, wasserstein_distance

# --- SETUP ---
app = Flask(__name__)
metrics = PrometheusMetrics(app)
accuracy_gauge = Gauge('model_accuracy', 'Current accuracy of the prediction model')
KS_gauge = Gauge('ks_statistic', 'Kolmogorov-Smirnov statistic')
Wasserstein_gauge = Gauge('wasserstein_distance', 'Wasserstein distance')

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:1234")
PREDICT_ENDPOINT = f"{MODEL_SERVER_URL}/invocations"

print(f"Flask app configured to make predictions to: {PREDICT_ENDPOINT}")

# --- METRIC CALCULATION LOGIC ---
def calculate_model_accuracy():
    """
    Loads data, sends each review separately to the model server for prediction,
    and returns the sample's accuracy.
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
            print(f"BACKGROUND: Sample accuracy calculated: {accuracy}")
            return accuracy
        else:
            print("BACKGROUND: All predictions failed; accuracy cannot be computed.")
            return 0.0

    except requests.exceptions.RequestException as e:
        print(f"BACKGROUND: Network error connecting to model server: {e}")
        return 0.0
    except Exception as e:
        print(f"BACKGROUND: An unexpected error occurred: {e}")
        return 0.0

def calculate_drift_metrics():
    try:
        DATA_DIR = 'Data'
        ref_data_path = os.path.join(DATA_DIR, 'test.csv')
        new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
        df_ref = pd.read_csv(ref_data_path)
        df_new = pd.read_csv(new_data_path)
        df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])
        df_new_latest = df_new.sort_values('Timestamp', ascending=False).head(100)
        ref_values = df_ref['Sentiment'].values
        new_values = df_new_latest['Sentiment'].values
        ks_stat, ks_p = ks_2samp(ref_values, new_values)
        wass_dist = wasserstein_distance(ref_values, new_values)
        return ks_stat, wass_dist
    except Exception as e:
        print(f"Error calculating drift metrics: {e}")
        return 0.0, 0.0

def calculate_all_metrics():
    accuracy = calculate_model_accuracy()
    ks_stat, wd_dist = calculate_drift_metrics()
    return accuracy, ks_stat, wd_dist

# Hitung metrik awal sebelum server Flask dijalankan
print("Menghitung metrik awal sebelum memulai server...")
initial_accuracy, initial_ks, initial_wd = calculate_all_metrics()
accuracy_gauge.set(initial_accuracy)
KS_gauge.set(initial_ks)
Wasserstein_gauge.set(initial_wd)
print(f"Metrik awal diatur: Akurasi={initial_accuracy}, KS={initial_ks}, WD={initial_wd}")

# --- BACKGROUND SCHEDULER ---
def background_scheduler():
    while True:
        accuracy, ks_stat, wd_dist = calculate_all_metrics()
        accuracy_gauge.set(accuracy)
        KS_gauge.set(ks_stat)
        Wasserstein_gauge.set(wd_dist)
        print(f"Scheduled update - Akurasi: {accuracy}, KS: {ks_stat}, WD: {wd_dist}")
        time.sleep(60)

NEW_FORM_HTML = '''
<!DOCTYPE html>
<html>
<head><title>Tambah Data Baru</title></head>
<body>
  <h2>Tambah Data Baru</h2>
  <form method="post">
    Review: <input type="text" name="review" required><br>
    Sentiment (0=Negative, 1=Positive): <input type="number" name="sentiment" min="0" max="1" required><br>
    <input type="submit" value="Submit">
  </form>
</body>
</html>
'''

@app.route('/new', methods=['GET', 'POST'])
def new_data():
    DATA_DIR = 'Data'
    new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
    if request.method == 'POST':
        review = request.form['review']
        sentiment = int(request.form['sentiment'])
        from datetime import datetime
        import pytz
        # Set timezone ke Indonesia (WIB = UTC+7)
        wib_tz = pytz.timezone('Asia/Jakarta')
        timestamp = datetime.now(wib_tz).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Adding new data with timestamp: {timestamp} (WIB)")
        
        # Optionally, call Gemini script here to generate synthetic review
        # For now, just append the submitted data
        header_exists = os.path.exists(new_data_path)
        with open(new_data_path, 'a', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            if not header_exists:
                writer.writerow(['Timestamp', 'Review', 'Sentiment'])
            writer.writerow([timestamp, review, sentiment])
        # Update metrics after adding new data
        ks_stat, wass_dist = calculate_drift_metrics()
        KS_gauge.set(ks_stat)
        Wasserstein_gauge.set(wass_dist)
        print(f"New data added. Updated KS: {ks_stat}, Wasserstein: {wass_dist}")
        return redirect(url_for('new_data'))
    return render_template_string(NEW_FORM_HTML)

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
