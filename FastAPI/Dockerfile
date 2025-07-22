# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN git config --global user.email "docker@example.com" && \
    git config --global user.name "Docker Container"

# Pre-download the embedding model to reduce startup time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the entire 'app' directory
COPY app/ ./app

# Copy the 'templates' directory for the HTML forms
COPY templates/ ./templates

# The config.py file now handles creating the Data directory
RUN mkdir -p /app/Data

# Expose port 8000 for the Uvicorn server
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
