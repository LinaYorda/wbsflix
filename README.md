# Movie Recommendation Application with Streamlit


Welcome to the WBSFLIX Movie Recommendation System project! This repository contains the code for a set of movie recommendation algorithms designed to power the online transition of WBSFLIX, a beloved DVD store based near Berlin. WBSFLIX, cherished for its local atmosphere and personalized movie recommendations by the owner,is embarking on a journey to bring its unique movie selection and personal touch to a wider audience online.

## Project Overview

As WBSFLIX expands, personalizing movie recommendations for every customer becomes a significant challenge. The goal of this project is to develop a series of recommender systems that mimic Ursula's knack for movie suggestions, ensuring that every user feels personally catered to, even in the digital space.

The project focuses on implementing various recommendation algorithms, ranging from basic collaborative filtering to more sophisticated machine learning models, to create a dynamic and personalized user experience similar to that of the physical store.

## Features

- **Your Personal Recommendations**: Users receive movie suggestions based on their viewing history, preferences, and ratings.
- **Our Top Rated Movies**: A section dedicated to what's popular among WBSFLIX users, showcasing the most-watched and highly-rated movies.


## Technologies Used

- **Python**: Primary programming language for algorithm development.
- **Pandas & NumPy**: For data manipulation and numerical calculations.
- **SciKit-Learn**: For implementing machine learning models.
- **TMDB API**: For images of the movies.
- **Streamlit**: For creating the web application interface.

## Algorithms for recommenders

In this project, we implement various strategies to curate personalized recommendations for users. Below are the approaches we've incorporated:

### Popularity-Based Recommendations

Popularity-based recommendation systems suggest items that are widely popular and highly rated by the majority. This method does not offer personalized recommendations but is effective in promoting items with broad appeal. It's particularly useful for new users (a phenomenon known as the cold start problem) or when personal data is sparse.

### User-Centric Recommendations (User-Based Collaborative Filtering)

User-Based Collaborative Filtering identifies similarities between users based on their ratings or interactions with items. By finding users with similar tastes or preferences, the system recommends items liked by one user to another, assuming that similar users will appreciate similar items. This approach personalizes recommendations but requires a sufficiently large dataset to find meaningful user connections.

### Item-Centric Recommendations (Item-Based Collaborative Filtering)

Item-Based Collaborative Filtering focuses on the relationships between items, recommending items similar to those a user has already liked or interacted with. It calculates similarities between items based on user ratings or interactions, assuming that users will likely be interested in items similar to their past preferences. This method is often more scalable and stable over time than user-based approaches, as item preferences change less frequently than user preferences.

## Acknowledgments

This project was made possible through the collaborative efforts of our team.
* [Karl Healy ](https://github.com/wheeliecopta)


