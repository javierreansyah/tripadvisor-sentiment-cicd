# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import gradio as gr
from skops import io as skops_io
import re

def preprocess_for_vectorizer(text):
    """Simple preprocessing function for the Vectorizer"""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"http", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)
    text = text.replace("@", " at ")
    text = text.lower()
    
    text = ' '.join(text.split())
    return text

pipeline = skops_io.load('Actions/Model/sentiment_pipeline.skops', trusted=[preprocess_for_vectorizer])

LABELS = ['Negative', 'Positive']

def predict_sentiment(text):
    proba = pipeline.predict_proba([text])[0]
    pred = pipeline.predict([text])[0]
    return {LABELS[0]: float(proba[0]), LABELS[1]: float(proba[1])}

description = "Enter a hotel review to get sentiment prediction (positive/negative)"

examples_data = [
    ["hated inn terrible, room-service horrible staff un-welcoming, decor recently updated lacks complete look, managment staff horrible.", "Negative"],
    ["best bar lobby meet friend year, pop elevator oliver great place drinks people watching, great location.", "Positive"],
    ["great room stay stayed nights business trip great hotel great room great food near.", "Positive"],
    ["beware beware leave vehicle, took advantage park ride unfortunately vehicle broken.", "Negative"],
    ["Bathroom was filthy with broken tiles and no hot water for three days straight.", "Negative"],
    ["Amazing breakfast buffet with ocean view, staff went above and beyond our expectations.", "Positive"],
    ["The room was dirty and smelled awful, AC didn't work and staff was rude.","Negative"],
    ["Perfect romantic getaway, beautiful spa facilities and delicious room service.","Positive"],
]

with gr.Blocks() as demo:
    iface = gr.Interface(
        fn=predict_sentiment,
        inputs=gr.Textbox(lines=4, label="Hotel Review"),
        outputs=gr.Label(num_top_classes=2, label="Sentiment Prediction"),
        title="TripAdvisor Hotel Review Sentiment Classification",
        description=description
    )
    
    gr.Markdown("## Input and Output Examples")
    gr.DataFrame(
        value=examples_data,
        headers=["Review Example", "Sentiment Prediction"],
        datatype=["str", "str"],
        interactive=False,
        wrap=True
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
