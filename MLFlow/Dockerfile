# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

FROM python:3.10-slim

WORKDIR /app

COPY MLFlow/requirements.txt .
RUN pip install -r requirements.txt

COPY MLFlow/ ./MLFlow/

CMD ["uvicorn", "MLFlow.app:app", "--host=0.0.0.0", "--port=5002"]
