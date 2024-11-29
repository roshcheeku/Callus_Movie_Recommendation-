import streamlit as st
import random

def apply_custom_styles():
    st.markdown("""
        <style>
            .stApp {
                background-image: url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&q=80');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }
            
            .stApp::before {
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.85);
                z-index: -1;
            }
            
            .stTextInput > div > div > input {
                background-color: rgba(41, 41, 41, 0.7) !important;
                color: white !important;
                border: 1px solid #404040 !important;
                padding: 1rem !important;
                font-size: 1.1rem !important;
                border-radius: 8px !important;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #9F7AEA !important;
                box-shadow: 0 0 0 1px #9F7AEA !important;
            }
            
            .movie-card {
                background-color: rgba(41, 41, 41, 0.7);
                padding: 1.5rem;
                border-radius: 10px;
                border: 1px solid #404040;
                margin-bottom: 1rem;
                transition: transform 0.2s;
            }
            
            .movie-card:hover {
                transform: translateY(-5px);
                border-color: #9F7AEA;
            }
            
            h1, h2, h3, p {
                color: white !important;
            }
            
            .stSpinner > div {
                border-color: #9F7AEA !important;
            }
            
            .stAlert {
                background-color: rgba(41, 41, 41, 0.7) !important;
                border: 1px solid #404040 !important;
            }
        </style>
    """, unsafe_allow_html=True)

def mock_get_recommendations(movie_name):
    # Simulated delay
    import time
    time.sleep(1)
    
    # Mock movie database
    movie_database = {
        "inception": ["The Matrix", "Interstellar", "Blade Runner 2049", "Source Code", "Tenet"],
        "the godfather": ["Goodfellas", "Casino", "Scarface", "The Departed", "Once Upon a Time in America"],
        "pulp fiction": ["Reservoir Dogs", "Kill Bill", "Django Unchained", "The Big Lebowski", "Snatch"],
        "titanic": ["The Notebook", "Pearl Harbor", "Romeo + Juliet", "Gone with the Wind", "Casablanca"]
    }
    
    # Return recommendations if movie exists, otherwise return random selection
    return movie_database.get(movie_name.lower(), random.sample([
        "The Shawshank Redemption", "Fight Club", "The Dark Knight",
        "Forrest Gump", "The Matrix", "Inception", "Pulp Fiction",
        "The Godfather", "Interstellar", "The Silence of the Lambs"
    ], 5))

def main():
    st.set_page_config(
        page_title="Movie Recommender",
        page_icon="🎬",
        layout="wide"
    )
    
    apply_custom_styles()
    
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 1rem; font-weight: bold;'>
                🎬 Movie Recommendations
            </h1>
            <p style='font-size: 1.2rem; color: #B8B8B8; max-width: 600px; margin: 0 auto;'>
                Discover new movies based on your favorites. Enter a movie title and we'll find similar films you might enjoy.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Search Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        movie_name = st.text_input(
            "",
            placeholder="Enter a movie name...",
            key="movie_search"
        )
    
    # Handle Search
    if movie_name:
        with st.spinner('Finding recommendations...'):
            try:
                recommendations = mock_get_recommendations(movie_name)
                
                # Display recommendations
                st.markdown("""
                    <h2 style='text-align: center; margin: 2rem 0; font-size: 1.8rem;'>
                        Recommended Movies
                    </h2>
                """, unsafe_allow_html=True)
                
                # Create grid layout
                cols = st.columns(3)
                for idx, (movie, col) in enumerate(zip(recommendations, cols * ((len(recommendations) + 2) // 3))):
                    with col:
                        st.markdown(f"""
                            <div class='movie-card'>
                                <p style='color: #9F7AEA !important; margin-bottom: 0.5rem; font-size: 0.9rem;'>
                                    Recommendation #{idx + 1}
                                </p>
                                <h3 style='margin: 0; font-size: 1.2rem;'>{movie}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error("An error occurred while fetching recommendations. Please try again.")
    
    # Initial state message
    if not movie_name:
        st.markdown("""
            <div style='text-align: center; margin-top: 2rem; padding: 2rem;
                        background-color: rgba(41, 41, 41, 0.7); border-radius: 10px;
                        border: 1px solid #404040; max-width: 600px; margin-left: auto;
                        margin-right: auto;'>
                <p style='color: #B8B8B8 !important; margin: 0;'>
                    Enter a movie title above to get personalized recommendations
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()