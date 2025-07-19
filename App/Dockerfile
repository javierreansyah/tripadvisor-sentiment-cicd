FROM python:3.10-slim

WORKDIR /app

COPY App/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY App/ ./App
COPY Model/ ./Model

EXPOSE 7860

CMD ["python", "App/app.py"]