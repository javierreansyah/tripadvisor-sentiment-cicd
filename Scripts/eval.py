# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import os
import pandas as pd
from skops import io as skops_io
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT_DIR, 'Model')
RESULTS_DIR = os.path.join(ROOT_DIR, 'Results')
DATA_DIR = os.path.join(ROOT_DIR, 'Data')

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model and vectorizer
model_path = os.path.join(MODEL_DIR, 'logreg_tfidf.skops')
vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.skops')
model = skops_io.load(model_path)
vectorizer = skops_io.load(vectorizer_path)

# Load data
csv_path = os.path.join(DATA_DIR, 'tripadvisor_hotel_reviews.csv')
df = pd.read_csv(csv_path)
df['Rating'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)
X = df['Review']
y = df['Rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform test data
X_test_tfidf = vectorizer.transform(X_test)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

metrics_path = os.path.join(RESULTS_DIR, 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, 'results.png')
plt.savefig(plot_path)
plt.close()
print(f"Confusion matrix plot saved to {plot_path}")