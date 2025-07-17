import asyncio
import csv
import os
import pytz
from datetime import datetime
from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import DATA_DIR
from app.services import calculate_drift_metrics
from app.metrics import KS_gauge, Wasserstein_gauge

router = APIRouter()

NEW_FORM_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Add New Data</title>
    <style>
        body { font-family: sans-serif; background-color: #f4f4f9; color: #333; margin: 2em; }
        h2 { color: #444; }
        form { background: white; padding: 2em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 500px; }
        input[type="text"], input[type="number"] { width: 100%; padding: 8px; margin-bottom: 1em; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        input[type="submit"] { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; }
        input[type="submit"]:hover { background-color: #0056b3; }
    </style>
</head>
<body>
  <h2>Add New Labeled Data</h2>
  <form method="post" action="/new">
    Review: <br>
    <input type="text" name="review" required><br>
    Sentiment (0=Negative, 1=Positive): <br>
    <input type="number" name="sentiment" min="0" max="1" required><br><br>
    <input type="submit" value="Submit">
  </form>
</body>
</html>
"""

@router.get("/new", response_class=HTMLResponse)
async def show_new_data_form():
    """Displays the HTML form to add new data."""
    return HTMLResponse(content=NEW_FORM_HTML)

@router.post("/new")
async def add_new_data(review: str = Form(...), sentiment: int = Form(...)):
    """Handles the submission of the new data form."""
    def _write_to_csv_sync():
        new_data_path = os.path.join(DATA_DIR, 'new_data.csv')
        wib_tz = pytz.timezone('Asia/Jakarta')
        timestamp = datetime.now(wib_tz).strftime("%Y-%m-%d %H:%M:%S")
        
        header_exists = os.path.exists(new_data_path) and os.path.getsize(new_data_path) > 0
        with open(new_data_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            if not header_exists:
                writer.writerow(['Timestamp', 'Review', 'Sentiment'])
            writer.writerow([timestamp, review, sentiment])

    await asyncio.to_thread(_write_to_csv_sync)

    ks_stat, wass_dist = await calculate_drift_metrics()
    KS_gauge.set(ks_stat)
    Wasserstein_gauge.set(wass_dist)
    print(f"New data added. Updated KS: {ks_stat:.4f}, Wasserstein: {wass_dist:.4f}")

    return RedirectResponse(url="/new", status_code=303)
