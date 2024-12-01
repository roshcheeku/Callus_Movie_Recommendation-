import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib

# Neo4j database connection details
uri = "neo4j+s://01405b8f.databases.neo4j.io"
username = "neo4j"
password = "3r0cFPqEYqgLVfaaEqv2-o9VIyiS_LZXgFAZNaew2BQ"
driver = GraphDatabase.driver(uri, auth=(username, password))

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
st.write("Fetching data from Neo4j...")
neo4j_data = fetch_movie_data()
st.write("Neo4j Data:\n", neo4j_data.head())

# Load additional CSV dataset
st.write("Loading CSV data...")
csv_data = pd.read_csv("cleaned_extracted_movie_genres.csv")
csv_data.rename(columns={"movie_name": "movie"}, inplace=True)
st.write("CSV Data:\n", csv_data.head())

# Merge the datasets on 'movie'
st.write("Merging datasets...")
merged_data = pd.merge(csv_data, neo4j_data, on="movie", how="inner")
st.write("Merged Data:\n", merged_data.head())

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

# Vectorize the genres, directors, and actors columns
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))

# Vectorize the features
genre_features = vectorizer.fit_transform(merged_data['genres'].fillna(""))
director_features = vectorizer.fit_transform(merged_data['directors'].fillna(""))
actor_features = vectorizer.fit_transform(merged_data['actors'].fillna(""))

# Combine features into a single matrix
combined_features = np.hstack([
    genre_features.toarray(),
    director_features.toarray(),
    actor_features.toarray()
])

# Train a kNN model for recommendations
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(combined_features)

# Save the trained kNN model
joblib.dump(knn, "knn_movie_recommender.pkl")

# Function to recommend movies
def recommend_movies(movie_name):
    try:
        st.write(f"Searching for '{movie_name}'...")
        # Normalize input to lowercase
        movie_name = movie_name.lower()
        
        # Find movie index
        movie_idx = merged_data[merged_data['movie'] == movie_name].index[0]
        st.write(f"Movie found: {merged_data.iloc[movie_idx]['movie']}")
        
        movie_vec = combined_features[movie_idx].reshape(1, -1)
        distances, indices = knn.kneighbors(movie_vec, n_neighbors=5)
        
        recommendations = merged_data.iloc[indices[0]]['movie'].tolist()
        st.write(f"Recommendations: {recommendations}")
        return recommendations
    except IndexError as e:
        st.write(f"Error: {e}")
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
