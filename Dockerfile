FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENTRYPOINT ["bash", "-c", "python download_data.py && uvicorn main:app --host 0.0.0.0 --port 8000"]