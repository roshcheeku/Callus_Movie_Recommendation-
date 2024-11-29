import pandas as pd
from neo4j import GraphDatabase

# Load the Bollywood dataset
df = pd.read_csv("cleaned_bollywood_data_set.csv")

# Print the column names to check for issues
print(df.columns)

# Clean up column names (if needed)
df.columns = df.columns.str.strip()

# Replace NaN values in 'actors' column with an empty string
df['actors'] = df['actors'].fillna('')

# Set up the Neo4j connection
uri = "bolt://localhost:7687"  # Replace with your Neo4j URI if different
username = "neo4j"  # Replace with your Neo4j username
password = "ROSH15VEDA"  # Replace with your Neo4j password

driver = GraphDatabase.driver(uri, auth=(username, password))

# Function to create nodes and relationships in Neo4j
def create_movie_data(tx, movie_name, year_of_release, runtime, IMDB_rating, no_of_votes, plot_description, actors):
    # Create movie node
    tx.run("""
        CREATE (m:Movie {name: $movie_name, year_of_release: $year_of_release, runtime: $runtime, 
                          IMDB_rating: $IMDB_rating, no_of_votes: $no_of_votes, plot_description: $plot_description})
        """, movie_name=movie_name, year_of_release=year_of_release, runtime=runtime, 
           IMDB_rating=IMDB_rating, no_of_votes=no_of_votes, plot_description=plot_description)
    
    # Create actor nodes and relationships
    for actor in actors.split(","):
        tx.run("""
            MERGE (a:Actor {name: $actor})
            MERGE (m)-[:ACTED_IN]->(a)
            """, actor=actor.strip())

# Insert the data into Neo4j
with driver.session() as session:
    for index, row in df.iterrows():
        session.write_transaction(create_movie_data, 
                                 row["movie_name"], row["year_of_release"], row["runtime"], row["IMDB_rating"], 
                                 row["no_of_votes"], row["plot_description"], row["actors"])

print("Data inserted into Neo4j!")
