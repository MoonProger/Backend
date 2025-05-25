import os
import json
import io
import pandas as pd
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

def download_to_bytesio(file_id: str) -> io.BytesIO:
    """
    Скачивает файл с Google Drive по file_id и возвращает BytesIO.
    """
    drive = get_drive_service()
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def stream_csv_chunks(file_id: str, chunksize: int = 100_000):
    """
    Возвращает итератор DataFrame-чанков из CSV на Google Drive.
    """
    fh = download_to_bytesio(file_id)
    # Передаём BytesIO как источник в pandas
    return pd.read_csv(fh, sep=',', chunksize=chunksize)