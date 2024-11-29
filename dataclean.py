import pandas as pd  

# Load the extracted movie genres data  
file_path = 'combined_bollywood_dataset.csv'  
df = pd.read_csv(file_path)  

# Display the initial DataFrame for reference  
print("Initial DataFrame:")  
print(df)  

# Check for missing values before processing  
print("\nMissing values before processing:")  
print(df.isnull().sum())  

# Handle NA values  
# Option 1: Replace NA with a placeholder  
df['genres'].fillna('Unknown', inplace=True)  # Replace NA values with 'Unknown'  

# Option 2: (commented out) You could also use this if you want to drop rows with NA values  
# df.dropna(inplace=True)  

# Display the DataFrame after handling missing values  
print("\nDataFrame after processing:")  
print(df)  

# Check for missing values after processing  
print("\nMissing values after processing:")  
print(df.isnull().sum())  

# Save the cleaned DataFrame to a new CSV file  
cleaned_file_path = 'cleaned_extracted_movie_genres.csv'  
df.to_csv(cleaned_file_path, index=False)  

print(f"\nCleaned data has been saved to '{cleaned_file_path}'.")