import streamlit as st
import pandas as pd
import requests
from surprise import Reader, Dataset, SVD
from dotenv import load_dotenv
import os

# Load environment variables for security
load_dotenv()
tmdb_api_key = os.getenv('TMDB_API_KEY')

# Fallback image URL for when a poster is not available
fallback_image_url = "https://via.placeholder.com/500x750?text=Poster+Not+Available"

st.set_page_config(
    page_title="WBSFLIX",
    page_icon=":shark:",
    layout="centered",
    initial_sidebar_state="expanded"
)

def set_custom_styles():
    st.markdown("""
        <style>
        /* Set the overall background color */
        .stApp {
            background-color: #FFFFFF; /* Light grey color */
        }
        
        /* Set the header color to red */
        h1, h2, h3 {
            color: #FF0000; /* Red color */
        }
        </style>
        """, unsafe_allow_html=True)

set_custom_styles()

# Continue with the rest of your app
#st.header("This is a red header")
#st.subheader("Subheaders are also red")
#st.write("The rest of the text will remain in default color.")

@st.cache_data
def load_data():
    ratings_path = 'ratings.csv'
    movies_path = 'movies.csv'
    links_path = 'links.csv'
    tags_path = "tags.csv"
    
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    links_df = pd.read_csv(links_path)
    tags_df = pd.read_csv(tags_path)
    
    links_df['movieId'] = links_df['movieId'].astype(movies_df['movieId'].dtype)
    movies_df = pd.merge(movies_df, links_df[['movieId', 'tmdbId']], on='movieId', how='left')
    merged_movies_ratings_df = pd.merge(movies_df, ratings_df, on="movieId")
    
    return merged_movies_ratings_df

@st.cache_data
def get_poster_url(tmdbId, api_key=tmdb_api_key):
    if pd.isna(tmdbId) or tmdbId == '':
        return fallback_image_url
    response = requests.get(f'https://api.themoviedb.org/3/movie/{tmdbId}?api_key={api_key}')
    if response.status_code == 200:
        data = response.json()
        return f'https://image.tmdb.org/t/p/w500{data.get("poster_path")}'
    return fallback_image_url

def display_movie_posters(section_title, movies_df):
    st.header(section_title)
    cols_per_row = 4
    rows_needed = len(movies_df) // cols_per_row + (len(movies_df) % cols_per_row > 0)
    for i in range(rows_needed):
        cols = st.columns(cols_per_row)
        for idx, col in enumerate(cols):
            movie_index = i * cols_per_row + idx
            if movie_index < len(movies_df):
                movie = movies_df.iloc[movie_index]
                poster_url = get_poster_url(movie['tmdbId'])
                col.image(poster_url, width=150, caption=movie['title'])

@st.cache_data
def top_n_movies(n, merged_movies_ratings_df):
    rating_summary = merged_movies_ratings_df.groupby('movieId').agg(average_rating=('rating', 'mean'), rating_count=('rating', 'count'))
    top_movies = pd.merge(merged_movies_ratings_df[['movieId', 'title', 'tmdbId']].drop_duplicates(), rating_summary, on='movieId')
    return top_movies.sort_values(by=['rating_count', 'average_rating'], ascending=[False, False]).head(n)

@st.cache_data
def item_based_recommendations(given_movieId, merged_data, n=6):
    from sklearn.metrics.pairwise import cosine_similarity

    user_movie_matrix = pd.pivot_table(merged_data,
                                       values='rating',
                                       index='userId',
                                       columns='movieId',
                                       fill_value=0)

    movies_cosines_matrix = pd.DataFrame(cosine_similarity(user_movie_matrix.T),
                                         index=user_movie_matrix.columns,
                                         columns=user_movie_matrix.columns)

    cosines_df = movies_cosines_matrix[[given_movieId]].sort_values(by=given_movieId, ascending=False).iloc[1:n+1]

    recommendations = cosines_df.merge(merged_data[['movieId', 'title', 'tmdbId']].drop_duplicates(), left_index=True, right_on='movieId').reset_index(drop=True)

    return recommendations

def setup_svd(merged_movies_ratings_df):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(merged_movies_ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD(random_state=123)
    algo.fit(trainset)
    return algo, trainset

@st.cache_data
def user_based_recommender(userId, n, _algo, _trainset, merged_movies_ratings_df):
    testset = _trainset.build_anti_testset()
    filtered_testset = [x for x in testset if x[0] == userId]
    predictions = _algo.test(filtered_testset)
    
    predictions_df = pd.DataFrame(predictions, columns=['userId', 'movieId', 'actual', 'est', 'details'])
    top_n_predictions = predictions_df.nlargest(n, 'est')
    
    top_n_movies_df = pd.merge(top_n_predictions, merged_movies_ratings_df[['movieId', 'title', 'tmdbId']], on='movieId', how='left').drop_duplicates(subset=['movieId']).head(n)
    
    return top_n_movies_df

# Main app logic
st.header("WBSFLIX")

# Initial data loading with spinner
with st.spinner('Please wait while we load the data...'):
    data = load_data()

# Setup SVD with spinner
with st.spinner('Setting up the recommendation engine...'):
    algo, trainset = setup_svd(data)

search_query = st.text_input("Search for movies:", "")

if search_query:
    with st.spinner('Searching for movies...'):
        search_results = data[data['title'].str.contains(search_query, case=False)]
        movie_id = search_results.iloc[0]['movieId'] if not search_results.empty else 1
else:
    movie_id = 1

with st.spinner('Fetching movie recommendations...'):
    recommendations = item_based_recommendations(movie_id, data, n=8)
    display_movie_posters("Your Personal Recommendations", recommendations)

top_movies = top_n_movies(8, data)
display_movie_posters("Our Top Rated Movies", top_movies)

user_id_input = st.number_input("Enter User ID for Recommendations", min_value=1, value=1, step=1)
number_of_recs = st.slider("Number of Recommendations", 1, 20, 8)

if st.button("Show Recommendations"):
    with st.spinner('Generating personalized recommendations...'):
        user_recs = user_based_recommender(user_id_input, number_of_recs, algo, trainset, data)
        if not user_recs.empty:
            display_movie_posters("Recommendations for You", user_recs)






