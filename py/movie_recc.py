import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

def compute_features(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_description'])
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['Genre'])
    combined_features = np.hstack((tfidf_matrix.toarray(), genre_matrix))
    return combined_features

def cosine_similarity_matrix(combined_features):
    return cosine_similarity(combined_features, combined_features)

def get_recommendations(title, cosine_sim, df):
    if title not in df['Title'].values:
        return ["Movie not found in the dataset."]
    
    idx = df[df['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]  # Exclude the first one (itself)

    recommended_titles = []
    recommended_movies = []
    seen_descriptions = set()

    for sim_idx, _ in sim_scores:
        row = df.iloc[sim_idx]
        if row['Description'] not in seen_descriptions and row['Title'] != title:
            if row['Description'] != df.iloc[idx]['Description']:  # Compare descriptions
                seen_descriptions.add(row['Description'])
                recommended_movies.append(row[['Title', 'Genre', 'Description', 'Director', 'Poster']])
            if len(recommended_movies) == 3:
                break

    return recommended_movies

def extract_imdb_id(imdb_link):
    imdb_id = re.search(r'tt\d+', imdb_link)
    if imdb_id:
        return imdb_id.group(0)
    return None

def fill_missing_info(df, imdb_id):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    api_url = f"http://www.omdbapi.com/?i={imdb_id}&apikey=7d9eb382"
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        movie_data = response.json()
        if movie_data['Response'] == 'True':
            new_movie = {
                'Title': movie_data.get('Title', 'N/A'),
                'Description': movie_data.get('Plot', 'N/A'),
                'Genre': movie_data.get('Genre', 'N/A').split(', '),
                'preprocessed_description': preprocess_text(movie_data.get('Plot', 'N/A'))
            }
            new_movie_df = pd.DataFrame([new_movie])
            df = pd.concat([df, new_movie_df], ignore_index=True)
    return df
