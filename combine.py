import pandas as pd

# Load the cleaned Bollywood dataset
bollywood_data = pd.read_csv('cleaned_bollywood_data_set.csv')

# Load the extracted genres dataset
extracted_genres = pd.read_csv('extracted_movie_genres.csv')

# Combine both datasets based on the movie_name column
combined_data = pd.merge(bollywood_data, extracted_genres, on='movie_name', how='inner')

# Save the combined dataset to a new CSV file
combined_data.to_csv('combined_bollywood_dataset.csv', index=False)

print("Datasets combined successfully! Saved as 'combined_bollywood_dataset.csv'.")
