from fastapi import FastAPI
import pandas as pd
import requests
import urllib.parse
import re
import json 
from pathlib import Path
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from googletrans import Translator
from fastapi import Query
from typing import List, Optional
# uvicorn main:app --reload --log-level debug
#lol
import os, io, pickle
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

app = FastAPI()

# Загружаем JSON-креды из переменной окружения
import json
creds_info = json.loads(os.environ['GOOGLE_SERVICE_ACCOUNT_JSON'])
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
drive = build('drive', 'v3', credentials=creds)

def download_file(file_id: str, dest: str):
    req = drive.files().get_media(fileId=file_id)
    fh = io.FileIO(dest, 'wb')
    dl = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        status, done = dl.next_chunk()
        print(f"{dest}: {int(status.progress()*100)}%")

# При старте — скачать всё нужное
@app.on_event("startup")
def load_data():
    files = {
        'ratings.csv':               'GOOGLE_DRIVE_FILE_ID_1',
        'fasttext_tfidf_cosine.pkl': 'GOOGLE_DRIVE_FILE_ID_2',
        # …другие файлы
    }
    for local, fid in files.items():
        if not os.path.exists(local):
            download_file(fid, local)
    # После скачивания можете один раз загрузить в память, если надо:
    # df_ratings = pd.read_csv('ratings.csv')
    # with open('fasttext_tfidf_cosine.pkl','rb') as f: model = pickle.load(f)

@app.get("/health")
def health():
    return {"status":"ok"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tough-seas-jump.loca.lt", "http://localhost:5173", "https://moviemap-inky.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("clusters_movies_with_tags.csv")
recommendations = pd.read_csv("recommendations.csv")
movies.set_index("movieId", inplace=True)

#для работы с API базы с фильмами
TMBD_API_KEY = "1482dc40dbcb47d03352529127eab8a1"
TMBD_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMBD_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

GENRE_TRANSLATIONS = {
    "Action": "Боевик",
    "Adventure": "Приключения",
    "Animation": "Анимация",
    "Children": "Детский",
    "Comedy": "Комедия",
    "Crime": "Криимнал",
    "Documentary": "Документальный",
    "Drama": "Драма",
    "Fantasy": "Фэнтези",
    "Film-Noir": "Фильм-нуар",
    "Horror": "Ужасы",
    "Musical": "Мюзикл",
    "Mystery": "Детектив",
    "Romance": "Мелодрама",
    "Sci-Fi": "Научная фантастика",
    "Thriller": "Триллер",
    "War": "Военный",
    "Western": "Вестерн",
    "(no genres listed)": "Не указано"
    
}

#кэширование информации о фильмах
cache_path = Path("movie_cache.json")
#кэширование похожих фильмов
similar_movies_path = Path("similar_movies_cache.json")

if cache_path.exists():
    with open(cache_path, "r", encoding="utf-8") as f:
        movie_cache = json.load(f)
        failed_cache = movie_cache.get("failed", [])
else:
    movie_cache = {}
    failed_cache = []

if similar_movies_path.exists():
    with open(similar_movies_path, "r", encoding="utf-8") as f:
        similar_cache = json.load(f)
else:
    similar_cache = {}
    
def save_cache():
    movie_cache["failed"] = list(set(failed_cache))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(movie_cache, f, ensure_ascii=False, indent=2)

def save_similar_movies_cache():
    with open(similar_movies_path, "w", encoding="utf-8") as f:
        json.dump(similar_cache, f, ensure_ascii=False, indent=2)
    
#извлечение данных о фильме с tmdb
def get_movie_info(movie_title: str, year: str, average_rating: float, movie_id: int) -> str:
    #пропускаем те, для которых не нашлась информация
    if movie_title in failed_cache:
        return None
    #проверяем не сохранен ли в кэше
    if str(movie_id) in movie_cache:
        return movie_cache[str(movie_id)]

    #преобразование названия фильма в формат для использования в url
    query = urllib.parse.quote(movie_title)
    #формирование url для запроса
    url = f"{TMBD_SEARCH_URL}?api_key={TMBD_API_KEY}&query={query}&year={year}&language=ru-RU"

    try:
        #отправление запроса и обработка ответа в формате json
        response = requests.get(url)
        data = response.json()
        if data["results"]:
            result = data["results"][0]

            #если нет рейтинга и он не извлекся -> неудачный
            if pd.isna(average_rating):
                tmdb_rating = result.get("vote_average", None)
                if tmdb_rating is None:
                    failed_cache.append(movie_title)
                    save_cache()
                    return None
                average_rating = tmdb_rating/2

            #извлекаем название, если не нашлось перевода -> неудачный
            ru_title = result.get("title", None)
            original_title = result.get("original_title", "")
            if (not ru_title or
                ru_title == original_title or not
                any(cyr in ru_title for cyr in "абвгддеёжзийклмнопрстуфхцчшщъыьэюя")):
                failed_cache.append(movie_title)
                save_cache()
                return None

            #извлекаем путь к постерау, если не нашлось -> неудачный
            poster_path = result.get("poster_path")
            if poster_path:
                poster_url = f"{TMBD_IMAGE_BASE_URL}{poster_path}"
            else:
                failed_cache.append(movie_title)
                save_cache()
                return None

            #извлекаем описание, если не нашлось -> неудачный
            overview = result.get("overview")
            original_overview = result.get("original_overview", "")
            if (not overview or
                overview == original_overview or not
                any(cyr in overview for cyr in "абвгддеёжзийклмнопрстуфхцчшщъыьэюя")):
                failed_cache.append(movie_title)
                save_cache()
                return None

            #переводим жанры, если не нашлось -> неудачный
            movie_row = movies.loc[movie_id]
            genres_raw = movie_row["genres"]
            if "(no genres listed)" in genres_raw or not genres_raw:
                failed_cache.append(movie_title)
                save_cache()
                return None
            genre_list = genres_raw.split("|")
            genre_names = [GENRE_TRANSLATIONS.get(g,g) for g in genre_list]

        #если не найдена информация -> неудачный
        else:
            failed_cache.append(movie_title)
            save_cache()
            return None

    except Exception as e:
        print(f"Ошибка при получении информации для фильма '{movie_title}': {e}")
        failed_cache.append(movie_title)
        save_cache()
        return None

    #формирование кэша
    movie_cache[str(movie_id)] = {
        "movie_id": int(movie_id),
        "ru_title": ru_title,
        "poster_url": poster_url,
        "overview": overview,
        "genres_ru": genre_names,
        "year": year,
        "average_rating": average_rating
    }
    save_cache()
    return movie_cache[str(movie_id)]

#обработка информации о фильме для извлечения данных с tmdb
def process_movie_info(row, movie_id):
    cleaned_title = re.sub(r'\(\d{4}\)|,.*$', '', row["title"]).strip()
    year_match = re.search(r'\((\d{4})\)', row["title"])
    year = year_match.group(1) if year_match else ''
    rating = row["avg_rating"] if "avg_rating" in row else row["rating"]
    info = get_movie_info(cleaned_title, year, round(rating, 1), movie_id)
    #если информации не хватает, то пропускаем
    if info is None:
        return None

    return{
        "movieId": int(movie_id),
        "title": info["ru_title"],
        "average_rating": info["average_rating"],
        "poster_url": info["poster_url"]
    }

#рекомендации на основе оценок пользователя
@app.get("/recommend/by-ratings/{user_id}")
def recommend_by_ratings(user_id: int, top_n: int=50):
    #получение оценок пользователя
    user_ratings = ratings[ratings["userId"]==user_id]
    if user_ratings.empty:
        return {"message": "Пользователь не оценивал фильмы"}
    #объединение оценок и фильмов по айди фильма
    merged = user_ratings.merge(movies, on="movieId", how="inner")
    #вычисление среднего рейтинга для каждого кластера по фильмам, которые смотрел пользователь
    cluster_ratings = merged.groupby("cluster")["rating"].mean().reset_index()
    #сортировка кластеров по рейтингу и выбор наилучшего кластера
    top_cluster = cluster_ratings.sort_values(by="rating", ascending=False).iloc[0]["cluster"]
    
    #получение фильмов из наилучшего кластера, которые пользователь не смотрел
    user_seen_movies = set(user_ratings["movieId"])
    cluster_movies = movies[movies["cluster"]==top_cluster]
    recommendations = cluster_movies[~cluster_movies.index.isin(user_seen_movies)].copy()
    
    #подставление среднего рейтинга фильмов
    recommendations["average_rating"] = recommendations["avg_rating"].fillna(0)
    #сортируем по убыванию среднего рейтинга
    sorted_recommendations = recommendations.sort_values(by="average_rating", ascending=False)
    
    result=[]
    #формирование ответа
    for movie_id, row in sorted_recommendations.iterrows():
        info = process_movie_info(row, movie_id)
        if info is None:
            continue
        result.append(info)
        if len(result) == top_n:
            break
    return {
        "recommendations": result
    }
    
#карточка фильма
@app.get("/movie/{movie_id}")
def get_movie_card(movie_id: int):
    movie_info = movie_cache[str(movie_id)]
    return {
        "movieId": int(movie_id),
        "title": movie_info["ru_title"],
        "poster_ur": movie_info["poster_url"],
        "average_rating": movie_info["average_rating"],
        "overview": movie_info["overview"],
        "year": movie_info["year"],
        "genres": movie_info["genres_ru"]
    }

#похожие фильмы
@app.get("/recommend/similar-movies/{movie_id}")
def recommend_similar_movies(movie_id: int, top_n: int = 10):
    in_cache = False
    
    if str(movie_id) in similar_cache:
        in_cache = True
        similar_indices = similar_cache[str(movie_id)]["indices"]
        movies_ft = joblib.load("fasttext_tfidf_cosine.pkl")[2]
    else:
        fasttext_model, tfidf, movies_ft = joblib.load("fasttext_tfidf_cosine.pkl")
        if movie_id not in movies_ft.index:
            return {"message": "у фильма нет тэгов, нельзя найти похожие"}
        #вектор исходного фильма
        movie_vector = movies_ft.loc[movie_id, "vector"].reshape(1, -1)
        #косинусное сходство с другими фильмами
        similarity_scores = cosine_similarity(movie_vector, np.vstack(movies_ft["vector"]))
        
        #top_n наиболее похожих фильмов
        similar_indices = []
        for pos in similarity_scores.argsort()[0][::-1]:
            current_id = movies_ft.index[pos]
            if current_id!=movie_id:
                similar_indices.append(int(current_id))
            if len(similar_indices) >= top_n*2:
                break
    
    result = []
    #формирование ответа
    for current_id in similar_indices:
        row = movies_ft.loc[current_id]
        info = process_movie_info(row, current_id)
        if info is None:
            continue
        result.append(info)
        if len(result) == top_n:
            break
        
    if not in_cache:
        similar_cache[str(movie_id)] = {
            "indices": [movie["movieId"] for movie in result]
        }
        save_similar_movies_cache()
        
    return {
        "recommendations": result
    }
    
#популярно у пользователей с похожим вкусом 
@app.get("/recommend/by-similar-ones/{user_id}")
def recommend_by_similar_ones(user_id: int, top_n: int = 50):
    #получение рекомендаций для пользователя
    user_recs = recommendations[recommendations['userId']== user_id]
    if user_recs.empty:
        return {"message": "Рекомендации для пользователя не найдены"}
    recs = user_recs['movieId'].tolist()
    movies_to_rec = movies[movies.index.isin(recs)].copy()
    sorted_recommendations = movies_to_rec.sort_values(by='avg_rating', ascending=False)
    
    result=[]
    #формирование ответа
    for movie_id, row in sorted_recommendations.iterrows():
        info = process_movie_info(row, movie_id)
        if info is None:
            continue
        result.append(info)
        if len(result) == top_n:
            break
    return {
        "recommendations": result
    }
    
#поиск по всем фильмам
@app.get("/movies/search")
async def search_movies(query: str, top_n: int = 20):
    #если параметр пустой - возвращаем пустой ответ
    if not query.strip():
        return {"results": []}
    
    result=[]
    
    #ищем по кэшу
    for movie_id, movie_data in movie_cache.items():
        if movie_id == "failed":
            continue
        if query.lower() in movie_data["ru_title"].lower():
            result.append({
                "movieId": int(movie_id),
                "title": movie_data["ru_title"],
                "average_rating": movie_data["average_rating"],
                "poster_url": movie_data["poster_url"]
            })
            if len(result)==top_n:
                return{"results": result}
    #если не набрались фильмы - переводим
    translator = Translator()
    translated = (await translator.translate(query, src='ru', dest='en')).text
    clean_query = re.sub(r'[^\w\s]', '', translated).strip()
    #ищем по переводу по всей базе без учета айди, что есть в кэше
    search_results = movies[
        (~movies.index.isin(movie_cache.keys())) & movies['title'].str.contains(rf'\b{clean_query}\b', case=False, na=False, regex=True)
    ]
    
    for movie_id, row in search_results.iterrows():
        info = process_movie_info(row, movie_id)
        if info is None:
            continue
        if query.lower() in info["title"].lower():
            result.append(info)
        if len(result) == top_n:
            break
    return {
        "results": result
    }
    
#фильтрация по жанру, рейтингу, году
def apply_filters(
    movies_ids: List[int], 
    genres: Optional[List[str]] = None, 
    min_rating: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None
):
    filtered = []
    for movie_id in movies_ids:
        movie_id = str(movie_id)
        movie = movie_cache[movie_id]
        if min_rating is not None and movie["average_rating"] < min_rating:
            continue
        if year_from is not None and int(movie["year"])<year_from:
            continue
        if year_to is not None and int(movie["year"])>year_to:
            continue
        if genres and not any(genre in movie["genres_ru"] for genre in genres):
            continue
        filtered.append({
            "movieId": int(movie_id),
            "title": movie["ru_title"],
            "average_rating": movie["average_rating"],
            "poster_url": movie["poster_url"]
        })
    return filtered

#фильтрация для "рекомендации на основе оценок пользователя"
@app.get("/recommend/by-ratings/{user_id}/filter")
async def filter_rating_reccomendations(
    user_id: int,
    genres: Optional[List[str]] = Query(None),
    min_rating: Optional[int] = Query(None, ge=1, le=5),
    year_from: Optional[int] = Query(None),
    year_to: Optional[int] = Query(None),
):
    recs = recommend_by_ratings(user_id)
    if "recommendations" not in recs:
        return recs
    movie_ids = [movie["movieId"] for movie in recs["recommendations"]]
    filtered = apply_filters(
        movie_ids,
        genres,
        min_rating,
        year_from,
        year_to
    )
    
    return {"recommendations": filtered}

#фильтрация для "популярно у пользователей с похожим вкусом"
@app.get("/recommend/by-similar-ones/{user_id}/filter")
async def filter_similar_reccomendations(
    user_id: int,
    genres: Optional[List[str]] = Query(None),
    min_rating: Optional[int] = Query(None, ge=1, le=5),
    year_from: Optional[int] = Query(None),
    year_to: Optional[int] = Query(None),
):
    recs = recommend_by_similar_ones(user_id)
    if "recommendations" not in recs:
        return recs
    movie_ids = [movie["movieId"] for movie in recs["recommendations"]]
    filtered = apply_filters(
        movie_ids,
        genres,
        min_rating,
        year_from,
        year_to
    )
    
    return {"recommendations": filtered}