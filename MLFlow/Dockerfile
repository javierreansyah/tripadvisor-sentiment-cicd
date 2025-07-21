FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY MLFlow/requirements.txt .
RUN pip install -r requirements.txt

# Copy MLFlow application files
COPY MLFlow/ ./MLFlow/

# Note: Data directory will be mounted as volume for real-time updates

# Set the default command to run the FastAPI app
CMD ["uvicorn", "MLFlow.app:app", "--host=0.0.0.0", "--port=5002"]
