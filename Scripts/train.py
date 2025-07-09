# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from skops import io as skops_io

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(ROOT_DIR, 'Data')
    MODEL_DIR = os.path.join(ROOT_DIR, 'Model')

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, 'tripadvisor_hotel_reviews.csv'))
    df['Rating'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)
    X = df['Review']
    y = df['Rating']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_tfidf, y_train)

    # Save model using SKOPS
    model_path = os.path.join(MODEL_DIR, 'logreg_tfidf.skops')
    skops_io.dump(model, model_path)

    # Save vectorizer
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.skops')
    skops_io.dump(vectorizer, vectorizer_path)