import pandas as pd
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK setup
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load the dataset
df = pd.read_csv("goodreads_data.csv")
df.drop(columns=["Unnamed: 0", "URL"], inplace=True)
df.dropna(inplace=True)
df["Genres"] = df["Genres"].str.split(", ").apply(lambda x: [genre.strip("[]") for genre in x])
df["Genres"] = df["Genres"].apply(lambda x: ', '.join(x))
df["Genres"] = df["Genres"].apply(lambda x: x.replace("'", ""))

def fetch_book_details(volume_id):
    url = f"https://www.googleapis.com/books/v1/volumes/{volume_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def extract_volume_id(url):
    match = re.search(r'/books/edition/.+/([^/?]+)', url)
    if match:
        return match.group(1)
    else:
        return None
    
def format_book_data(book_data):
    volume_info = book_data.get("volumeInfo", {})
    book = volume_info.get("title", "N/A")
    authors = ", ".join(volume_info.get("authors", ["N/A"]))
    description = volume_info.get("description", "N/A")
    genres = ", ".join(volume_info.get("categories", ["N/A"]))
    avg_rating = volume_info.get("averageRating", "N/A")
    
    book_dict = {
        "Book": book,
        "Author": authors,
        "Description": description,
        "Genres": genres,
        "Avg_Rating": avg_rating
    }
    
    return book_dict

def search_google_books(book_title, author):
    query = f"{book_title} {author}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("items", [])
        if results:
            volume_info = results[0]["volumeInfo"]
            info_link = volume_info.get("infoLink", "N/A")
            cover_url = volume_info["imageLinks"].get("thumbnail", "N/A") if "imageLinks" in volume_info else "N/A"
            return cover_url, info_link
    return "N/A", "N/A"

def preprocess_text(text):
    if pd.isnull(text):  # Check if the text is NaN
        return ""        # If NaN, return an empty string
    
    # Tokenize, lemmatize, and remove stopwords
    tokens = word_tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]
    return " ".join(clean_tokens)

def compute_similarity(descriptions, genres):
    # Vectorize text
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_desc = tfidf_vectorizer.fit_transform(descriptions)
    tfidf_matrix_genres = tfidf_vectorizer.fit_transform(genres)

    # Compute cosine similarity
    description_similarity = cosine_similarity(tfidf_matrix_desc)
    genre_similarity = cosine_similarity(tfidf_matrix_genres)

    return description_similarity, genre_similarity

def find_recommendations(input_book_idx, book_data, description_similarity, genre_similarity):
    combined_similarity = (description_similarity[input_book_idx] + genre_similarity[input_book_idx]) / 2

    similar_indices = combined_similarity.argsort()[-4:-1][::-1]
    similar_books = [(book_data.iloc[idx]["Book"], book_data.iloc[idx]["Author"]) for idx in similar_indices]

    return similar_books

def main():
    url = input("Paste your Google Books URL here: ")
    volume_id = extract_volume_id(url)
    book_data = fetch_book_details(volume_id)
    formatted_data = format_book_data(book_data)

    input_book = formatted_data["Book"]
    input_description = preprocess_text(formatted_data["Description"])
    input_genres = preprocess_text(formatted_data["Genres"])

    # Add input book's data to the dataframe using pd.concat
    input_book_df = pd.DataFrame([{
        "Book": input_book,
        "Description": input_description,
        "Genres": input_genres,
        "Author": formatted_data["Author"]
    }])
    
    df_extended = pd.concat([df, input_book_df], ignore_index=True)

    # Compute similarity
    description_similarity, genre_similarity = compute_similarity(df_extended["Description"], df_extended["Genres"])

    # Find recommendations
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

if __name__ == "__main__":
    main()