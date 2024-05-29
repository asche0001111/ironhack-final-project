import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re
import os
#import env
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Initialize Spotipy with Spotify API credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="5fcb240462b148c8b565a4323950392b",
    client_secret="7bf80eadadb8437883db4ab82e463876"
))

def extract_track_id(spotify_url):
    match = re.match(r'(https?://open\.spotify\.com/track/|spotify:track:)([a-zA-Z0-9]+)', spotify_url)
    if match:
        return match.group(2)
    else:
        raise ValueError("Invalid Spotify URL")
    
def get_track_details(track_id):
    track = sp.track(track_id)
    features = sp.audio_features(track_id)
    album_art_url = track['album']['images'][0]['url']  # Extract album art URL from track information
    return {
        'id': track_id,
        'name': track['name'],
        'artist': track['artists'][0]['name'],
        'artist_id': track['artists'][0]['id'],
        'album': track['album']['name'],
        'release_date': track['album']['release_date'],
        'energy': features[0]['energy'],
        'tempo': features[0]['tempo'],
        'valence': features[0]['valence'],
        'url': track['external_urls']['spotify'],
        'album_art_url': album_art_url  # Include album art URL in the dictionary
    }

def get_related_artists(artist_id):
    related_artists = sp.artist_related_artists(artist_id)
    return [artist['id'] for artist in related_artists['artists']]

def get_artist_top_tracks(artist_id):
    top_tracks = sp.artist_top_tracks(artist_id)
    return [track['id'] for track in top_tracks['tracks']]

def get_related_top(related_artists):
    related_top_tracks = []
    for i in related_artists:
        top_tracks = sp.artist_top_tracks(i)
        related_top_tracks += [track['id'] for track in top_tracks['tracks']]
    return related_top_tracks

def get_related_features(top_artist_rel):
    related_features = []
    for i in top_artist_rel:
        track = sp.track(i)
        features = sp.audio_features(i)
        album_art_url = track['album']['images'][0]['url']  # Extract album art URL from track information
        related_features.append({
            'id': i,
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'artist_id': track['artists'][0]['id'],
            'album': track['album']['name'],
            'release_date': track['album']['release_date'],
            'energy': features[0]['energy'],
            'tempo': features[0]['tempo'],
            'valence': features[0]['valence'],
            'url': track['external_urls']['spotify'],
            'album_art_url': album_art_url  # Include album art URL in the dictionary
        })
    return related_features

def recommend_tracks(track_data, track_features, kmeans_model, n_recommendations=3):
    input_features = np.array([[track_data['energy'], track_data['tempo'], track_data['valence']]])
    input_cluster = kmeans_model.predict(input_features)[0]
    
    cluster_indices = [i for i, label in enumerate(kmeans_model.labels_) if label == input_cluster]
    
    cluster_tracks = [track_features[i] for i in cluster_indices]
    cluster_features = np.array([[f['energy'], f['tempo'], f['valence']] for f in cluster_tracks])
    
    distances = euclidean_distances(input_features, cluster_features)[0]
    closest_indices = distances.argsort()[:n_recommendations + 1]  # Get one extra to account for input track
    
    recommended_tracks = [cluster_tracks[i] for i in closest_indices if cluster_tracks[i]['id'] != track_data['id']]
    
    return recommended_tracks[:n_recommendations]

def main():
    spotify_url = input("Input your Spotify track URL here: ")
    track_id = extract_track_id(spotify_url)
    track_data = get_track_details(track_id)
    artist_id = track_data['artist_id']
    related_artists = get_related_artists(artist_id)
    top_artist_rel = get_related_top(related_artists)
    track_features = get_related_features(top_artist_rel)
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

if __name__ == "__main__":
    main()