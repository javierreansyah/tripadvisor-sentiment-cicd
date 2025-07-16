FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY Flask/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The NLTK downloader line is removed as this service no longer uses NLTK directly.

# Copy app code
COPY Flask/app.py .

# Make port 5000 available
EXPOSE 5000

# Run the app
CMD ["flask", "run", "--host=0.0.0.0"]
