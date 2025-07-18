# Use a modern, slim Python base image
FROM python:3.10-slim

# Install system dependencies for DVC and git
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Configure git (required for DVC)
RUN git config --global user.email "docker@example.com" && \
    git config --global user.name "Docker Container"

# Download NLTK data required for the model's preprocessor
RUN python -m nltk.downloader stopwords wordnet punkt

# Copy the entire 'app' directory (containing your modular code)
COPY app/ ./app

# Copy the 'templates' directory for the HTML forms
COPY templates/ ./templates

# The config.py file now handles creating the Data directory
RUN mkdir -p /app/Data

# Expose port 8000 for the Uvicorn server
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
