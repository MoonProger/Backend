#FastText + TF-IDF + cosine similarity
import pandas as pd
from gensim.models import FastText
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

movies = pd.read_csv("clusters_movies_with_tags.csv")
#пропускаем пустые тэги
movies = movies[movies["tag"].notna() & (movies["tag"].str.strip() != "")]
def normalize_tag(tag_str):
    tag_str = tag_str.lower().strip() #нижний регистр и удаление пробелов
    tag_str = re.sub(r'[-_]', ' ', tag_str) #замена дефисов и подчеркиваний на пробелы
    tag_str = re.sub(r'[^\w\s]', '', tag_str) #удаление всех символов кроме букв, цифр и пробелов
    tag_str = re.sub(r'\s+', ' ', tag_str) #замена многих пробелов на один пробел
    return tag_str
movies["tags"] = movies["tag"].apply(normalize_tag)

#установка айди фильмов в индексы
movies.set_index("movieId", inplace=True)

def identity_tokenizer(text):
    return text

def train_model():
    import os
    os.makedirs("models", exist_ok=True)
    
    #токенизация тэгов и обучение модели
    movies["tags_tokens"] = movies["tags"].apply(lambda x: x.split())
    model = FastText(sentences=movies["tags_tokens"], vector_size=150, window=3, min_count=3, workers=4, epochs=100)

    #векторизация
    tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, token_pattern=None)
    tfidf_matrix = tfidf.fit_transform(movies["tags_tokens"])
    idf_weights = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    #усреднение векторов fasttext с весами tf-idf
    def get_weighted_vector(tokens):
        vectors = []
        weights = []
        for token in tokens:
            if token in model.wv and token in idf_weights:
                vectors.append(model.wv[token] * idf_weights[token])
                weights.append(idf_weights[token])
        if vectors:
            return np.average(vectors, axis=0, weights=weights)
        else:
            return np.zeros(model.vector_size)
    movies["vector"] = movies["tags_tokens"].apply(get_weighted_vector)
    
    joblib.dump((model, tfidf, movies), "models/fasttext_tfidf_cosine.pkl")

def get_similar_movies(movie_id: int, top_n: int = 10):
    model, tfidf, movies = joblib.load("models/fasttext_tfidf_cosine.pkl")
    #вектор исходного фильма
    movie_vector = movies.loc[movie_id, "vector"].reshape(1, -1)
    
    #косинусное сходство с другими фильмами
    similarity_scores = cosine_similarity(movie_vector, np.vstack(movies["vector"]))
    
    #top_n наиболее похожих фильмов
    similar_indices = similarity_scores.argsort()[0][::-1][1:top_n + 1]
    similar_movies = movies.iloc[similar_indices][["title", "tags"]].copy()
    similar_movies["similarity"] = similarity_scores[0][similar_indices]
    similar_movies.reset_index(inplace=True)
    
    return similar_movies.to_dict(orient="records")

#пример: находим похожие на movieId=1
if __name__ == "__main__":
    #train_model()
    similar = get_similar_movies(movie_id=1997, top_n=10)
    for s in similar:
        print(f"{s['movieId']}, {s['title']} — Сходство: {s['similarity']:.3f}")