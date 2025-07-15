FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY Flask/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the required NLTK data package
RUN python -m nltk.downloader punkt # <-- ADD THIS LINE

# Copy app code
COPY Flask/app.py .

# Copy the model directory into the container's /app/Model directory
COPY Model/ /app/Model

# Make port 5000 available
EXPOSE 5000

# Run the app
CMD ["flask", "run", "--host=0.0.0.0"]