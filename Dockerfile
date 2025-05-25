FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py gdrive_utils.py clusters_movies_with_tags.csv movie_cache.json similar_movies_cache.json./
# только нужные скрипты

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]