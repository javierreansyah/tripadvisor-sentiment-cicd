# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import os
import nltk
import pandas as pd
from skops import io as skops_io
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT_DIR, 'Model')
RESULTS_DIR = os.path.join(ROOT_DIR, 'Results')
DATA_DIR = os.path.join(ROOT_DIR, 'Data')

os.makedirs(RESULTS_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, 'logreg_tfidf.skops')
vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.skops')
model = skops_io.load(model_path)
vectorizer = skops_io.load(vectorizer_path, trusted=[nltk.tokenize.word_tokenize])

test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
X_test = test_df['Cleaned_Review']
y_test = test_df['Sentiment']

X_test_tfidf = vectorizer.transform(X_test)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print(report)

metrics_path = os.path.join(RESULTS_DIR, 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(report)
print(f"Metrics saved to {metrics_path}")

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