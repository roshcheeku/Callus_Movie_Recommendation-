import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib
import zipfile
import os

# Neo4j database connection details
uri = "bolt://localhost:7687"
username = "neo4j"
password = "ROSH15VEDA"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Extract the kNN model from the zip file
MODEL_ZIP_PATH = "knn_movie_recommender.pkl.zip"
MODEL_FILE_PATH = "knn_movie_recommender.pkl"

if not os.path.exists(MODEL_FILE_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall()  # Extracts in the current directory

# Load the pre-trained kNN model
knn = joblib.load(MODEL_FILE_PATH)

# Function to fetch movie data from Neo4j
def fetch_movie_data():
    query = """
    MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
    OPTIONAL MATCH (m)-[:DIRECTED]->(d:Director)
    OPTIONAL MATCH (m)-[:ACTED_IN]->(a:Actor)
    RETURN m.name AS movie, 
           COLLECT(DISTINCT g.name) AS genres,
           COLLECT(DISTINCT d.name) AS directors,
           COLLECT(DISTINCT a.name) AS actors;
    """
    with driver.session() as session:
        result = session.run(query)
        data = []
        for record in result:
            data.append({
                "movie": record["movie"],
                "genres": ",".join(record["genres"]),
                "directors": ",".join(record["directors"]),
                "actors": ",".join(record["actors"])
            })
    return pd.DataFrame(data)

# Fetch movie data
print("Fetching data from Neo4j...")
neo4j_data = fetch_movie_data()
print("Neo4j Data:\n", neo4j_data.head())

# Load additional CSV dataset
print("Loading CSV data...")
csv_data = pd.read_csv("cleaned_extracted_movie_genres.csv")
csv_data.rename(columns={"movie_name": "movie"}, inplace=True)
print("CSV Data:\n", csv_data.head())

# Merge the datasets on 'movie'
print("Merging datasets...")
merged_data = pd.merge(csv_data, neo4j_data, on="movie", how="inner")
print("Merged Data:\n", merged_data.head())

# Handle duplicate columns for genres
if "genres_x" in merged_data.columns and "genres_y" in merged_data.columns:
    merged_data["genres"] = merged_data["genres_x"]
    merged_data.drop(columns=["genres_x", "genres_y"], inplace=True)

# Ensure 'actors' column exists
if "actors_x" in merged_data.columns or "actors_y" in merged_data.columns:
    if "actors" not in merged_data.columns:
        merged_data["actors"] = merged_data.get("actors_x", merged_data.get("actors_y"))
    merged_data.drop(columns=["actors_x", "actors_y"], inplace=True, errors="ignore")

# Normalize movie names to lowercase in the merged dataset
merged_data['movie'] = merged_data['movie'].str.lower()

# Combine features into a single matrix
def combine_features(data):
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
    genre_features = vectorizer.fit_transform(data['genres'].fillna(""))
    director_features = vectorizer.fit_transform(data['directors'].fillna(""))
    actor_features = vectorizer.fit_transform(data['actors'].fillna(""))
    return np.hstack([genre_features.toarray(), director_features.toarray(), actor_features.toarray()])

combined_features = combine_features(merged_data)

# Function to recommend movies
def recommend_movies(movie_name):
    try:
        # Normalize input to lowercase
        movie_name = movie_name.lower()
        
        # Find movie index
        movie_idx = merged_data[merged_data['movie'] == movie_name].index[0]
        movie_vec = combined_features[movie_idx].reshape(1, -1)
        distances, indices = knn.kneighbors(movie_vec, n_neighbors=5)
        recommendations = merged_data.iloc[indices[0]]['movie'].tolist()
        return recommendations
    except IndexError:
        return ["Movie not found in the dataset!"]

# Streamlit UI
def main():
    st.title("Movie Recommendation System")
    st.write("This is a movie recommendation system based on genres, directors, and actors.")

    movie_name = st.text_input("Enter a movie name for recommendations:")

    if movie_name:
        recommendations = recommend_movies(movie_name)
        if isinstance(recommendations, list):
            st.write(f"Recommended movies for '{movie_name}':")
            for idx, rec in enumerate(recommendations, 1):
                st.write(f"{idx}. {rec}")
        else:
            st.write(recommendations)

if __name__ == "__main__":
    main()
