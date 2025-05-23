FROM python:3.11-slim

WORKDIR /app

# Копируем файл зависимостей и ставим их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем только код (без больших датасетов)
COPY main.py .

# Указываем команду запуска
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]