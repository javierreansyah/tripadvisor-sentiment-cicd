# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import gradio as gr
import pandas as pd
from skops import io as skops_io
from sklearn.feature_extraction.text import TfidfVectorizer
import os

model = skops_io.load('Model/logreg_tfidf.skops')
vectorizer = skops_io.load('Model/tfidf_vectorizer.skops')

LABELS = ['Negative', 'Positive']

def predict_sentiment(text):
    X_tfidf = vectorizer.transform([text])
    proba = model.predict_proba(X_tfidf)[0]
    pred = model.predict(X_tfidf)[0]
    return {LABELS[0]: float(proba[0]), LABELS[1]: float(proba[1])}

description = "Masukkan review hotel untuk mendapatkan prediksi sentimen (positif/negatif)"

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, label="Review Hotel"),
    outputs=gr.Label(num_top_classes=2, label="Prediksi Sentimen"),
    title="Klasifikasi Sentimen Review Hotel TripAdvisor",
    description=description
)

iface.launch(server_name="0.0.0.0", server_port=7860)