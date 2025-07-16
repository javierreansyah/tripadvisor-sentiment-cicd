FROM python:3.10-slim

WORKDIR /app

# Copy requirements from the MLFlow directory and install
COPY MLFlow/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet punkt

# Copy the entire project context
COPY . .

# Set the default command to run the script from its new location
CMD ["python", "MLFlow/run_training.py"]
