import pickle
import pandas as pd
import streamlit as st

# Load the data
movies_data = pickle.load(open('movie_list_update.pkl', mode='rb'))
data = pd.DataFrame(movies_data)

similarity = pickle.load(open('similarity.pkl', mode='rb'))

# Function to recommend movies
def recommend(movie):
    try:
        index = data[data['Series_Title'] == movie].index[0]
        distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommend_title = [data.iloc[i[0]].Series_Title for i in distance[1:8]]
        recommend_poster = [data.iloc[i[0]].Poster_Link for i in distance[1:8]]
        return recommend_title, recommend_poster
    except IndexError:
        print(f"Movie '{movie}' not found in the dataset.") #Consider logging this instead of printing
        return [], []

# Add intro text to sidebar
st.sidebar.markdown("# About Me")
intro_text = """
👋 Hi there!  
I'm **Pratham Gupta**, and I’m thrilled to share my latest **Machine Learning** project: the **Movie Recommendation System**.  
This system uses smart algorithms like **content-based filtering** to provide personalized movie suggestions based on user preferences.  

### Curious About the Code?  
The full implementation is available on my GitHub, complete with documentation to help you dive right in.  
👉 [Visit My GitHub](https://github.com/Code1235/Machine-Learning-Projects/tree/main/Movies%20Recommendation%20System)  

Thanks for stopping by, and happy exploring!
"""
st.sidebar.markdown(intro_text)

# Streamlit Web-App
st.title("Movie Recommendation System")
selected_movie = st.selectbox("Select a movie:", data['Series_Title'].values)
btn = st.button("Show Recommendations")

if btn:
    movie_titles, movie_posters = recommend(selected_movie)
    if movie_titles:
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        
        with col1:
            st.write(movie_titles[0])
            st.image(movie_posters[0])
        with col2:
            st.write(movie_titles[1])
            st.image(movie_posters[1])
        with col3:
            st.write(movie_titles[2])
            st.image(movie_posters[2])
        with col4:
            st.write(movie_titles[3])
            st.image(movie_posters[3])
        with col5:
            st.write(movie_titles[4])
            st.image(movie_posters[4])
        with col6:
            st.write(movie_titles[5])
            st.image(movie_posters[5])
        with col7:
            st.write(movie_titles[6])
            st.image(movie_posters[6])
    else:
        st.subheader("No recommendations found for this movie.")