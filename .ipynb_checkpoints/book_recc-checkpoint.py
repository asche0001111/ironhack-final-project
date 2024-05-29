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

import os
print(os.getcwd())

# Load the dataset
df = pd.read_csv("goodreads_data.csv")
df.drop(columns=["Unnamed: 0", "URL"], inplace=True)
df.dropna(inplace=True)
df["Genres"] = df["Genres"].str.split(", ").apply(lambda x: [genre.strip("[]") for genre in x])
df["Genres"] = df["Genres"].apply(lambda x: ', '.join(x))
df["Genres"] = df["Genres"].apply(lambda x: x.replace("'", ""))

print(df.head())