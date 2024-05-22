solo final project for IronHack

built using the SpotiPy API, IMDb dataset from Kaggle and Open Library API

Interactive media suggestion model with a web interface

MOVING PARTS:
PART 1: SPOTIFY SONG SUGGESTION ALGORITHM
- Using the SpotiPy python library to access the Spotify API
- Paste the Spotify link in the search bar, retrieve related artists as well as numerical properties of a given track (IE: Energy, Tempo…)
- Have an unsupervised ML algorithm analyze the track and give back (one to five, still undecided) track(s) NOT by the same artist
- Have an option to see the related tracks of the suggested output, making it so that the list is infinitely cyclable as long as there are related tracks to suggest

PART 2: MOVIE/TV SHOW SUGGESTION ALGORITHM
- Either use an IMDB movies dataset from Kaggle or scrape it all from the website 
- Paste an IMDB movie link in the search bar, retrieve related media via natural language processing of the description
- Have an option to see related movies/tv shows from the output
  
PART 3: BOOK SUGGESTION ALGORITHM
- Still need to figure out how to actually get the data for processing, whether it is a kaggle dataset or scraping an API 
- Have a similar natural language processing model trained as for the movie recommendations, going off of the summary of the books
- Have an option to see the related books of the output

PART 4: MAKE IT FANCY
- Have basic HTML/CSS in place to give this whole program a user interface
- Have 3 query boxes, one for the music, the other for the movies and the other for the books that the user pastes the Spotify/imdb link into
- maybe with a “How it works” button at the bottom that explains how the system works
