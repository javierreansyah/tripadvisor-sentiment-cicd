# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import os
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from skops import io as skops_io

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
MODEL_DIR = os.path.join(ROOT_DIR, 'Model')
os.makedirs(MODEL_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

X_train = train_df['Cleaned_Review']
y_train = train_df['Sentiment']


vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, tokenizer=word_tokenize)
X_train_tfidf = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_tfidf, y_train)

skops_io.dump(model, os.path.join(MODEL_DIR, 'logreg_tfidf.skops'))
skops_io.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.skops'))

print(f"Model and vectorizer saved to {MODEL_DIR}")

