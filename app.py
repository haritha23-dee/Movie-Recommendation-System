import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("top10K-TMDB-movies.csv")

# Fill missing values
movies["genre"] = movies["genre"].fillna("")
movies["overview"] = movies["overview"].fillna("")

# Combine genre and overview for better recommendations
movies["combined_features"] = movies["genre"] + " " + movies["overview"]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create mapping from movie title to index
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Function to get similar movies
def recommend_movies(title, num_recommendations=5):
    if title not in indices:
        return ["Movie not found. Try another title."]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies["title"].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.subheader("Find similar movies based on genre & overview")

# Dropdown for movie selection
movie_list = movies["title"].dropna().unique()
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Get Recommendations"):
    recommendations = recommend_movies(selected_movie)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")

