import pickle
import pandas as pd
import streamlit as st


# Load the data

data = pickle.load(open('data_dict.pkl', mode='rb'))
data = pd.DataFrame(data)
# print(data)

similarity = pickle.load(open('similarity.pkl', mode='rb'))
# print(similarity)


def recommend(movie):
    
    recommended = []
    
    movie_index = data[data['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
    
    for i in movie_list:
        movie_id = i[0]
        # fetch poster

        recommended.append(data.iloc[i[0]].title)

    return recommended



# Streamlit Web-App

st.title(':white[Movie Recommendation System]')
selected_movie = st.selectbox('Select your movie: ', data['title'].values)
btn = st.button('Recommend')

if btn:
    recommended_movies = recommend(selected_movie)

    for i in recommended_movies:
        st.write(i)

        