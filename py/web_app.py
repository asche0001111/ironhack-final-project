import sys
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, render_template, request, redirect, url_for
from spotify_recc import authenticate_spotify, extract_track_id, get_track_details, get_related_artists, get_related_top, get_related_features, recommend_tracks
from movie_recc import preprocess_text as preprocess_movie_text, compute_features, cosine_similarity_matrix, get_recommendations, extract_imdb_id, fill_missing_info
from book_recc import fetch_book_details, extract_volume_id, format_book_data, preprocess_text as preprocess_book_text, compute_similarity, find_recommendations, search_google_books

app = Flask(__name__)

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app.template_folder = template_dir

def authenticate_spotify_with_form(client_id, client_secret):
    return authenticate_spotify(client_id, client_secret)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_link', methods=['POST'])
def process_link():
    if request.method == 'POST':
        link = request.form['link']
        link
        return redirect(url_for('index'))

@app.route('/recommend', methods=['POST'])
def recommend():
    spotify_client_id = request.form['spotify_client_id']
    spotify_client_secret = request.form['spotify_client_secret']
    sp = authenticate_spotify_with_form(spotify_client_id, spotify_client_secret)
    link = request.form['link']
    recommendations = []
    recommendations_title = ""

    # Check if the link is a Spotify link
    if "open.spotify.com" in link or "spotify:track:" in link:
        if sp is None:
            error_message = "Spotify authentication failed. Please check your credentials and try again."
            return render_template('index.html', error_message=error_message)
        
        spotify_url = link
        track_id = extract_track_id(spotify_url)
        track_data = get_track_details(sp, track_id)
        artist_id = track_data['artist_id']
        related_artists = get_related_artists(sp, artist_id)
        top_artist_rel = get_related_top(sp, related_artists)
        track_features = get_related_features(sp, top_artist_rel)
        X = np.array([[f['energy'], f['tempo'], f['valence']] for f in track_features])
        kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

        recommended_tracks = recommend_tracks(track_data, track_features, kmeans)
        recommendations_title = f"Recommendations for '{track_data['name']}' by {track_data['artist']}:"
        for track in recommended_tracks:
            recommendations.append({
                'title': track['name'],
                'album': track['album'],
                'artist': track['artist'],
                'album_art_url': track['album_art_url'],
                'url': track['url']
            })
    
    # Check if the link is an IMDb link
    elif "imdb.com/title/tt" in link:
        df = pd.read_csv("imdb-movies-dataset.csv")
        df['Genre'] = df['Genre'].fillna('')
        df['Genre'] = df['Genre'].astype(str)
        df['Genre'] = df['Genre'].apply(lambda x: x.split(', '))

        imdb_id = extract_imdb_id(link)
        if imdb_id:
            df = fill_missing_info(df, imdb_id)
            if not df.empty:
                df['preprocessed_description'] = df['Description'].apply(preprocess_movie_text)
                combined_features = compute_features(df)
                cosine_sim = cosine_similarity_matrix(combined_features)
                movie_title = df.iloc[-1]['Title']
                recommendations_title = f"Recommendations for '{movie_title}':"
                recommendations_data = get_recommendations(movie_title, cosine_sim, df)
                for movie in recommendations_data:
                    recommendations.append({
                        'title': movie['Title'],
                        'description': movie['Description'],
                        'genre': movie['Genre'],
                        'director': movie['Director'],
                        'poster': movie['Poster']
                    })
            else:
                recommendations_title = "Failed to retrieve movie information from IMDb."
        else:
            recommendations_title = "Invalid IMDb link. Please provide a valid IMDb link."
    
    # Check if the link is a Google Books link
    elif "google" and "books" in link:
        df = pd.read_csv("goodreads_data.csv")
        df.drop(columns=["Unnamed: 0", "URL"], inplace=True)
        df.dropna(inplace=True)
        df["Genres"] = df["Genres"].str.split(", ").apply(lambda x: [genre.strip("[]") for genre in x])
        df["Genres"] = df["Genres"].apply(lambda x: ', '.join(x))
        df["Genres"] = df["Genres"].apply(lambda x: x.replace("'", ""))

        volume_id = extract_volume_id(link)
        book_data = fetch_book_details(volume_id)
        formatted_data = format_book_data(book_data)

        input_book = formatted_data["Book"]
        recommendations_title = f"Recommendations for '{input_book}':"
        input_description = preprocess_book_text(formatted_data["Description"])
        input_genres = preprocess_book_text(formatted_data["Genres"])

        input_book_df = pd.DataFrame([{
            "Book": input_book,
            "Description": input_description,
            "Genres": input_genres,
            "Author": formatted_data["Author"]
        }])

        df_extended = pd.concat([df, input_book_df], ignore_index=True)
        description_similarity, genre_similarity = compute_similarity(df_extended["Description"], df_extended["Genres"])
        input_book_idx = df_extended.index[df_extended["Book"] == input_book][0]
        recommendations_data = find_recommendations(input_book_idx, df_extended, description_similarity, genre_similarity)

        for book, author in recommendations_data:
            cover_url, book_url = search_google_books(book, author)
            book_row = df_extended[df_extended['Book'] == book].iloc[0]
            recommendations.append({
                'title': book,
                'author': author,
                'description': book_row['Description'],
                'genres': book_row['Genres'],
                'cover_url': cover_url,
                'book_url': book_url
            })

    else:
        recommendations_title = "Unsupported link format. Please provide a Spotify track URL, an IMDb link, or a Google Books URL."

    return render_template('index.html', recommendations=recommendations, recommendations_title=recommendations_title)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
