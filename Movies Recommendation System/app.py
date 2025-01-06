import pickle
import pandas as pd
import streamlit as st


# Load the data
movies_data = pickle.load(open('movie_list_update.pkl', mode='rb'))
data = pd.DataFrame(movies_data)

similarity = pickle.load(open('similarity.pkl', mode='rb'))


def recommend(movie):
    recommended = []
    index = data[data['Series_Title'] == movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    
    for i in distance[1:6]:
        recommended.append({
            'title': data.iloc[i[0]].Series_Title,
            'poster': data.iloc[i[0]].Poster_Link
        })

    return recommended

# Add intro text to upper left corner
st.sidebar.markdown("# About me:")

intro_text = """
Hi!👋 \n
I'm Pratham, and this is my latest Machine Learning project.\n
Also, you will find information on most searched movies on this site by users, as well as daily user traffic\n 
If you're curious about the code and want to explore it, feel free to visit my Github account! [GitHub](https://github.com/Code1235)\n
"""
st.sidebar.markdown(intro_text)

# Streamlit Web-App
st.title('Movie Recommendation System')
selected_movie = st.selectbox('Select your movie: ', data['Series_Title'].values)
btn = st.button('Recommend')

if btn:
    recommended_movies = recommend(selected_movie)

    for movie in recommended_movies:
        st.subheader(movie['title'])
        st.image(movie['poster'])
