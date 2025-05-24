FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем оба скрипта
COPY download_data.py main.py ./

# Запускаем сначала download_data.py, потом Uvicorn без reload и с одним worker
ENTRYPOINT ["bash", "-c", "python download_data.py && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1"]