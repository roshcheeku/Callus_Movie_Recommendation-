#not implented 
import pandas as pd
import requests
from neo4j import GraphDatabase

# Load the cleaned Bollywood dataset (adjust path as needed)
df = pd.read_csv('cleaned_bollywood_data_set.csv')

# Neo4j connection setup
uri = "bolt://localhost:7687"
username = "neo4j"
password = "ROSH15VEDA"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Function to extract genres from the OMDB API using IMDb ID
def extract_genres_from_api(movie_id):
    # OMDB API URL with the IMDb ID parameter
    api_url = f"http://www.omdbapi.com/?i={movie_id}&apikey=3b03da8e"  # Replace with your actual API key
    response = requests.get(api_url)
    
    if response.status_code == 200:
        movie_data = response.json()
        # Check if the 'Genre' key exists in the response
        genres = movie_data.get('Genre', '').split(', ') if 'Genre' in movie_data else []
        return genres
    else:
        print(f"Failed to fetch data for movie ID {movie_id}.")
        return []

# Function to update the Neo4j database with genres for a movie
def update_movie_genres_in_neo4j(movie_name, genres):
    with driver.session() as session:
        for genre in genres:
            session.run("""
                MERGE (m:Movie {name: $movie_name})
                MERGE (g:Genre {name: $genre})
                MERGE (m)-[:HAS_GENRE]->(g)
            """, movie_name=movie_name, genre=genre)

# Function to update the CSV file with extracted genres for each movie
def update_genre_csv():
    extracted_genres = []
    for idx, row in df.iterrows():
        movie_id = row['imdb-id']  # Correct column name to 'imdb-id'
        movie_name = row['movie_name']
        genres = extract_genres_from_api(movie_id)
        extracted_genres.append((movie_name, genres))
        print(f"Extracted genres for {movie_name}: {genres}")

        # Update Neo4j with genres for the current movie
        update_movie_genres_in_neo4j(movie_name, genres)

    # Save the extracted genres to a CSV file
    genre_df = pd.DataFrame(extracted_genres, columns=['movie_name', 'genres'])
    genre_df.to_csv('extracted_movie_genres.csv', index=False)
    print("Genres have been successfully extracted and saved to 'extracted_movie_genres.csv'.")

# Initialize TF-IDF Vectorizer and transform plot descriptions into numerical features
from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer(stop_words='english').fit_transform(df['plot_description'])

# Initialize the KNN model (using cosine distance to measure similarity)
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=3, metric='cosine')
knn.fit(X)

# Function to recommend movies based on plot descriptions (KNN)
def knn_recommend(movie_name):
    movie_idx = df[df['movie_name'] == movie_name].index[0]
    movie_vec = X[movie_idx]
    distances, indices = knn.kneighbors(movie_vec, n_neighbors=3)
    
    recommended_movies = df.iloc[indices[0]]['movie_name'].tolist()
    return recommended_movies

# Neo4j-based recommendation function
def recommend_movies_by_features(movie_name):
    query = """
    MATCH (m:Movie)-[:ACTED_IN]->(a:Actor), 
          (m)-[:HAS_GENRE]->(g:Genre), 
          (m)-[:DIRECTED_BY]->(d:Director)
    WHERE m.name = $movie_name
    WITH COLLECT(a) AS actors, COLLECT(g) AS genres, d AS director
    MATCH (other:Movie)-[:ACTED_IN]->(a:Actor), 
          (other)-[:HAS_GENRE]->(g:Genre), 
          (other)-[:DIRECTED_BY]->(d:Director)
    WHERE other.name <> $movie_name AND (a IN actors OR g IN genres OR d = director)
    WITH other, COUNT(DISTINCT a) AS actor_score, COUNT(DISTINCT g) AS genre_score, 
         CASE WHEN d = director THEN 1 ELSE 0 END AS director_score
    RETURN other.name AS recommended_movie, 
           (actor_score + genre_score + director_score) AS score
    ORDER BY score DESC
    LIMIT 3;
    """
    # Set up Neo4j connection
    with driver.session() as session:
        result = session.run(query, movie_name=movie_name)
        recommendations = [record["recommended_movie"] for record in result]
    
    return recommendations

# Example combined function for getting recommendations
def combined_recommendations(movie_name):
    # First, get recommendations based on Neo4j and KNN
    graph_based_recommendations = recommend_movies_by_features(movie_name)
    knn_based_recommendations = knn_recommend(movie_name)
    
    # Combine both lists and remove duplicates
    combined_recs = list(set(graph_based_recommendations + knn_based_recommendations))
    
    return combined_recs

# Example usage
if __name__ == "__main__":
    # Update the genres and save to the CSV file and Neo4j
    update_genre_csv()
    
    # Allow the user to input a movie name for recommendations
    movie_name = input("Enter the name of the movie you want recommendations for: ")
    recommended_movies = combined_recommendations(movie_name)
    
    print(f"Recommended movies for '{movie_name}': {recommended_movies}")