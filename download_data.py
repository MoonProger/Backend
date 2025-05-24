import os
import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def get_drive_service():
    creds_info = json.loads(os.environ['GOOGLE_SERVICE_ACCOUNT_JSON'])
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build('drive', 'v3', credentials=creds)

def download_file(drive, file_id: str, dest_path: str):
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
    drive = get_drive_service()
    files = {
        'ratings.pkl':               '1LXcpFzOXzr-5XG24-FLEHUQpZqeEHTEb',
        'fasttext_tfidf_cosine.pkl': '1zpE7LH9jpUy7C8CuckIJECxwqqk0IXT9',
        'clusters_movies_with_tags.csv': '1pWBK57DsDHK1eskGP7Sn6tMaUjtUIWlB',
        'movie_cache.json': '1oXzttp3KLszbA7Y5ganda4Xh2eKfolY5',
        'recommendations.csv': '10AcCRQY3bodl0wwJJmPWJTSakRQH6rnY',
        'similar_movies_cache.json': '1rNArkV1xoOsuPNAxxs1j20VHEo6Z_BxR',
    }

    for local_name, file_id in files.items():
        if os.path.exists(local_name):
            print(f"✅ {local_name} уже существует, пропускаем загрузку")
        else:
            download_file(drive, file_id, local_name)

    print("🎉 Все файлы загружены!")

if __name__ == '__main__':
    if 'GOOGLE_SERVICE_ACCOUNT_JSON' not in os.environ:
        raise RuntimeError("Не найдена переменная GOOGLE_SERVICE_ACCOUNT_JSON")
    main()