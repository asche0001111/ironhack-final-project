import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from spotify_recc import authenticate_spotify, extract_track_id, get_track_details, get_related_artists, get_related_top, get_related_features, recommend_tracks
from movie_recc import preprocess_text as preprocess_movie_text, compute_features, cosine_similarity_matrix, get_recommendations, extract_imdb_id, fill_missing_info
from book_recc import fetch_book_details, extract_volume_id, format_book_data, preprocess_text as preprocess_book_text, compute_similarity, find_recommendations, search_google_books

def main():
    link = input("Please paste a Spotify track URL, an IMDb link, or a Google Books URL: ")

    # Check if the link is a Spotify link
    if "open.spotify.com" in link or "spotify:track:" in link:
        sp = authenticate_spotify()
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
        print(f"Recommendations for '{track_data['name']}' by {track_data['artist']}:")
        for track in recommended_tracks:
            print(f"Title: {track['name']}\n"
                  f"Album: {track['album']}\n"
                  f"Artist: {track['artist']}\n"
                  f"Cover: {track['album_art_url']}\n"
                  f"URL: {track['url']}\n")
    
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
                recommendations = get_recommendations(df.iloc[-1]['Title'], cosine_sim, df)
                print("Recommendations:")
                for movie in recommendations:
                    print(f"Title: {movie['Title']}")
                    print(f"Description: {movie['Description']}")
                    print(f"Genre: {', '.join(movie['Genre'])}")
                    print(f"Director: {movie['Director']}")
                    print(f"Poster: {movie['Poster']}")
                    print()
            else:
                print("Failed to retrieve movie information from IMDb.")
                sys.exit(1)
        else:
            print("Invalid IMDb link. Please provide a valid IMDb link.")
            sys.exit(1)
    
    # Check if the link is a Google Books link
    elif "books/edition/" in link:
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
        recommendations = find_recommendations(input_book_idx, df_extended, description_similarity, genre_similarity)

        print(f"Recommendations for '{input_book}':")
        for book, author in recommendations:
            cover_url, book_url = search_google_books(book, author)
            book_row = df_extended[df_extended['Book'] == book].iloc[0]
            print(f"Title: {book}")
            print(f"Author: {author}")
            print(f"Description: {book_row['Description']}")
            print(f"Genres: {book_row['Genres']}")
            print(f"Cover: {cover_url}")
            print(f"Link: {book_url}")
            print()

    else:
        print("Unsupported link format. Please provide a Spotify track URL, an IMDb link, or a Google Books URL.")
        sys.exit(1)

if __name__ == "__main__":
    main()
