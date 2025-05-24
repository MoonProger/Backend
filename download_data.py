import pandas as pd
import pickle
import os
import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def get_drive_service():
    """
    Читает из env-переменной GOOGLE_SERVICE_ACCOUNT_JSON и
    создаёт объект Google Drive API.
    """
    creds_info = json.loads(os.environ['GOOGLE_SERVICE_ACCOUNT_JSON'])
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build('drive', 'v3', credentials=creds)

def download_file(drive, file_id: str, dest_path: str):
    """
    Скачивает файл с указанным file_id в локальный путь dest_path.
    """
    print(f"⬇️  Start downloading {dest_path} (id={file_id})")
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(dest_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"   {dest_path}: {int(status.progress() * 100)}%")
    print(f"✅  Finished {dest_path}\n")

def main():
    # 1) Инициализация сервиса
    drive = get_drive_service()

    # 2) Словарь: локальное имя → Google Drive file ID
    files = {
        'ratings.csv':               '1_Nwrnq3UuIwlQnay-pWrzRql7glHVsCp',
        'fasttext_tfidf_cosine.pkl': '1zpE7LH9jpUy7C8CuckIJECxwqqk0IXT9',
        # если есть ещё файлы, добавьте их сюда:
        # 'clusters_movies_with_tags.csv': 'ВАШ_FILE_ID',
        # 'movie_cache.json': 'ВАШ_FILE_ID',
        # 'recommendations.csv': 'ВАШ_FILE_ID',
        # 'similar_movies_cache.json': 'ВАШ_FILE_ID',
    }

    # 3) Скачиваем по одному, если нет на диске
    for local_name, file_id in files.items():
        if not os.path.exists(local_name):
            download_file(drive, file_id, local_name)
        else:
            print(f"ℹ️  {local_name} уже существует, пропускаем.")

    print("🎉 Все файлы загружены!")

if __name__ == '__main__':
    # Проверим, что переменная окружения задана
    if 'GOOGLE_SERVICE_ACCOUNT_JSON' not in os.environ:
        raise RuntimeError("Не найдена переменная GOOGLE_SERVICE_ACCOUNT_JSON")
    main()