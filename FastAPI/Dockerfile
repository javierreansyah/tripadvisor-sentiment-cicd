# Use a modern, slim Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file from the build context (./FastAPI) and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required for the model's preprocessor
RUN python -m nltk.downloader stopwords wordnet punkt

# Copy the entire 'app' directory (containing your modular code) into the container
COPY app/ ./app

# The config.py file now handles creating the Data directory,
# so the RUN mkdir command is no longer strictly necessary but can be kept for clarity.
RUN mkdir -p /app/Data

# Expose port 8000 for the Uvicorn server
EXPOSE 8000

# Command to run the FastAPI application using the new entry point.
# Uvicorn will look for the 'app' object inside the 'app.main' module.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
