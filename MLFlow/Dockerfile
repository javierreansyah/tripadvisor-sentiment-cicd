FROM python:3.10-slim

WORKDIR /app

# Copy requirements from the MLFlow directory and install
COPY MLFlow/requirements.txt .
RUN pip install -r requirements.txt

# Copy the entire project context
COPY . .

# Set the default command to run the script from its new location
CMD ["python", "MLFlow/run_training.py"]
