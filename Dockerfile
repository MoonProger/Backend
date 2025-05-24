FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_data.py main.py ./

# Загрузка файлов + небольшая пауза + запуск FastAPI
ENTRYPOINT ["bash", "-c", "python download_data.py && sleep 3 && uvicorn main:app --host 0.0.0.0 --port 8000"]