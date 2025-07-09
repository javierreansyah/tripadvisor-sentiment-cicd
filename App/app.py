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

description = "Enter a hotel review to get sentiment prediction (positive/negative)"

# Contoh data untuk tabel
examples_data = [
    ["The hotel was amazing! Great service and beautiful rooms.", "Positive"],
    ["Terrible experience. The room was dirty and staff was rude.", "Negative"],
    ["Good location but the breakfast could be better.", "Negative"],
    ["Worst hotel ever. Would not recommend to anyone.", "Negative"],
    ["Nice hotel with comfortable beds and friendly staff.", "Positive"]
]

with gr.Blocks() as demo:
    # Interface utama
    iface = gr.Interface(
        fn=predict_sentiment,
        inputs=gr.Textbox(lines=4, label="Hotel Review"),
        outputs=gr.Label(num_top_classes=2, label="Sentimen Prediction"),
        title="TripAdvisor Hotel Review Sentiment Classification",
        description=description
    )
    
    # Tabel contoh
    gr.Markdown("## Input and Output Examples")
    gr.DataFrame(
        value=examples_data,
        headers=["Review Example", "Sentiment Prediction"],
        datatype=["str", "str"],
        interactive=False,
        wrap=True
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)