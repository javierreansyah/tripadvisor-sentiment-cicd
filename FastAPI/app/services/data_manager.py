import os
import csv
import pytz
import json
import google.generativeai as genai
from datetime import datetime

from app.config import DATA_DIR

# --- Data Handling & Metrics (Unchanged) ---
def write_data_to_csv(data_to_write: list):
    new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
    header_exists = os.path.exists(new_data_path) and os.path.getsize(new_data_path) > 0
    with open(new_data_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        if not header_exists:
            writer.writerow(['Timestamp', 'Review', 'Sentiment'])
        writer.writerows(data_to_write)

def add_manual_data(review: str, sentiment: int):
    wib_tz = pytz.timezone('Asia/Jakarta')
    timestamp = datetime.now(wib_tz).strftime("%Y-%m-%d %H:%M:%S")
    write_data_to_csv([[timestamp, review, sentiment]])
    print("New manual data added. Metrics will be updated on the next scheduled run.")

def generate_and_save_gemini_data(style: str, quantity: int):
    prompt = f"""
    Generate a list of {quantity} realistic hotel reviews.
    The style of the reviews must be: {style}.
    For each review, provide a sentiment score: 1 for a positive review and 0 for a negative review.
    Ensure each review is concise and under 80 words.
    Include a mix of both positive and negative experiences.
    """
    try:
        review_schema = {
            "type": "ARRAY", "items": {
                "type": "OBJECT", "properties": {
                    "Review": {"type": "STRING"}, "Sentiment": {"type": "INTEGER"}
                }, "required": ["Review", "Sentiment"]
            }
        }
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json", response_schema=review_schema
        )
        model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash-latest", generation_config=generation_config
        )
        response = model.generate_content(prompt)
        reviews_data = json.loads(response.text)
        wib_tz = pytz.timezone('Asia/Jakarta')
        timestamp = datetime.now(wib_tz).strftime("%Y-%m-%d %H:%M:%S")
        data_to_write = [[timestamp, item['Review'], item['Sentiment']] for item in reviews_data]
        write_data_to_csv(data_to_write)
        return len(reviews_data)
    except Exception as e:
        raise e
