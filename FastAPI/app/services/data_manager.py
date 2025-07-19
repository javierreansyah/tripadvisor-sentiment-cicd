import os
import pytz
import json
import random
import pandas as pd
import google.generativeai as genai
from datetime import datetime

from app.config import DATA_DIR

def get_random_examples_from_gemini_csv(num_examples: int = 5):
    """Get random examples from gemini_example.csv file using pandas"""
    gemini_example_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'gemini_example.csv')
    examples = []
    
    if os.path.exists(gemini_example_path):
        # Read CSV with pandas
        df = pd.read_csv(gemini_example_path)
        
        # Filter out empty reviews and get the Review column
        reviews = df['Review'].dropna().tolist()
        
        # Get random samples
        if len(reviews) >= num_examples:
            examples = random.sample(reviews, num_examples)
        else:
            examples = reviews
    return examples

def write_data_to_csv(data_to_write: list):
    new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
    df_new = pd.DataFrame(data_to_write, columns=['Timestamp', 'Review', 'Sentiment'])
    if os.path.exists(new_data_path) and os.path.getsize(new_data_path) > 0:
        df_new.to_csv(new_data_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df_new.to_csv(new_data_path, mode='w', header=True, index=False, encoding='utf-8')

def generate_and_save_gemini_data(style: str = None, quantity: int = 20):
    # Get 5 random sample reviews from existing gemini_example.csv for reference
    sample_reviews = get_random_examples_from_gemini_csv(5)
    
    style_instruction = f"The style of the reviews must be: {style}." if style else "Use a natural, authentic hotel review style similar to real TripAdvisor reviews."
    
    # Create examples string from random samples
    examples_text = "\n".join([f'- "{review}"' for review in sample_reviews])
    
    prompt = f"""
    Generate a list of {quantity} realistic hotel reviews that match the style and format of real TripAdvisor reviews.
    {style_instruction}
    For each review, provide a sentiment score: 1 for a positive review and 0 for a negative review.
    Make the reviews detailed and authentic, similar to these examples:
    
    Examples:
    {examples_text}
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
            model_name="models/gemini-2.5-flash", generation_config=generation_config
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
