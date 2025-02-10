import pandas as pd

# Load the dataset
movies = pd.read_csv("top10K-TMDB-movies.csv")

# Display the first few rows
print(movies.head())

# Get information about the dataset
print(movies.info())

# Check for missing values in the dataset
print(movies.isnull().sum())

# Fill missing values in 'genre' and 'overview' with an empty string
movies["genre"] = movies["genre"].fillna("")
movies["overview"] = movies["overview"].fillna("")

# Check again for missing values
print(movies.isnull().sum())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer (removes common words)
tfidf = TfidfVectorizer(stop_words="english")

# Transform 'genre' column into a numerical matrix
tfidf_matrix = tfidf.fit_transform(movies["genre"])

# Print the shape of the TF-IDF matrix
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print the shape of the similarity matrix
print(cosine_sim.shape)

# Create a mapping from movie titles to index
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Function to get similar movies
def recommend_movies(title, num_recommendations=5):
    # Get the index of the movie
    idx = indices[title]

    # Get similarity scores for all movies with this movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar movies (excluding the first one, which is the same movie)
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top recommended movie titles
    return movies["title"].iloc[movie_indices]

#Example: Get recommendations for a movie
print(recommend_movies("The Godfather"))

