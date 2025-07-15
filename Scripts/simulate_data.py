from dotenv import load_dotenv
import google.generativeai as genai
import json
import os
import pandas as pd
import csv

load_dotenv()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    if not API_KEY:
        raise ValueError("API key not found.")


genai.configure(api_key=API_KEY)

review_schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "Review": {
                "type": "STRING",
                "description": "A hotel review, must be under 80 words."
            },
            "Sentiment": {
                "type": "INTEGER",
                "description": "The sentiment of the review: 0 for negative, 1 for positive."
            }
        },
        "required": ["Review", "Sentiment"]
    }
}

generation_config = genai.GenerationConfig(
    response_mime_type="application/json",
    response_schema=review_schema
)

model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash-lite-preview-06-17",
    generation_config=generation_config
)

prompt = """
Generate a list of 10 realistic hotel reviews.
For each review, provide a sentiment score: 1 for a positive review and 0 for a negative review.
Ensure each review is concise and under 80 words.
Include a mix of both positive and negative experiences.
"""

try:
    print("Generating hotel reviews... Please wait.")
    response = model.generate_content(prompt)
    reviews_data = json.loads(response.text)

    print("Saving data to Data/new_data.csv...")
    df = pd.DataFrame(reviews_data)
    
    csv_file = os.path.join(DATA_DIR, 'new_data.csv')

    file_exists = os.path.exists(csv_file)

    df.to_csv(csv_file, mode='a', header=not file_exists, index=False, quoting=csv.QUOTE_NONNUMERIC)

    if not file_exists:
        print(f"'{csv_file}' created and data saved.")
    else:
        print(f"Data appended to existing '{csv_file}'.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure your API key is correct and has the necessary permissions.")