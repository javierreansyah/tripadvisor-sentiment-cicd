FROM python:3.10-slim

WORKDIR /app

COPY App/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY App/ ./App
COPY Actions/Model/ ./Actions/Model

EXPOSE 7860

CMD ["python", "App/app.py"]