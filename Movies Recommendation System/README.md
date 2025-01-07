# Movie Recommendation System

## Introduction 

<br><br>
Welcome to the Movie Recommendation System! This project uses Machine Learning to suggest movies based on user input. Whether you’re looking for something similar to your favorite film or just exploring new options, this system has you covered. 🎥✨🍿
<br><br>

---

Explore my site Here [Visit](https://movie-recommendation-system-hpwawwnergnkwqk4ajdnkb.streamlit.app/)

---

## Features

🖋️ Personalized Recommendations: Provides movie suggestions based on user preferences.<br><br>

🖋️ Content-Based Filtering: Suggests similar movies by analyzing their features.<br><br>

🖋️ Streamlit Web App: User-friendly interface for selecting and viewing recommendations. 🎯📽️🤖<br><br>

---

## Technologies Used <br><br>

🖋️ Programming Language: Python<br><br>

&nbsp;&nbsp;   📃 Libraries:<br><br>

&nbsp;&nbsp;&nbsp;&nbsp;       📖 Pandas<br><br>

 &nbsp;&nbsp;&nbsp;&nbsp;      📖 Streamlit<br><br>

 &nbsp;&nbsp;&nbsp;&nbsp;      📖 Pickle<br><br>

🖋️ Machine Learning: Content-based filtering with cosine similarity 🎛️📊💻<br><br>

---

## Dataset <br><br>

The dataset includes movie titles, genres, descriptions, and poster links. It has been preprocessed and saved as a pickle file for efficient loading. 📂🎬🗂️ <br><br>

Dataset Link: [Visit](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
<br><br>

---

## Installation
<br><br>
Follow these steps to set up the project on your local machine: 🚀🛠️🔧<br><br>


1.Clone the repository:<br><br>

```bash
git clone https://github.com/Code1235/Machine-Learning-Projects
cd Movies-Recommendation-System
```

<br>
2.Install the required Python libraries:<br><br>

```bash
pip install -r requirements.txt
```
<br>

3.Ensure the following files are in your project directory:<br><br>

`movie_list_update.pkl`: Contains preprocessed movie data.<br><br>

`similarity.pkl`: Contains the precomputed similarity matrix.<br><br>

---

## How to Run<br><br>

Start the Streamlit web app:<br><br>

```bash
streamlit run app.py
```
<br><br>
Open the provided local URL in your web browser (e.g., `http://localhost:8501`).<br><br>

Use the dropdown to select a movie, then click on Show Recommendations to view the suggestions along with their posters. 🎬💻🌟<br><br>

---

## Functionality<br><br>

Core Function: `recommend`<br><br>

The recommendation system uses the following approach: 🧠🔍🎞️<br><br>

- Finds the index of the selected movie in the dataset.<br><br>

- Computes the cosine similarity with other movies.<br><br>

- Retrieves the top 5 similar movies and their poster links.<br><br>

### Example Output:

For the input movie "Harry Potter and the Deathly Hallows: Part 1", the recommendations might include: 🌀🎥🎭<br><br>

Harry Potter and the Deathly Hallows: Part 2<br><br>

Harry Potter and the Prisoner of Azkaban
<br><br>
Harry Potter and the Half-Blood Prince	<br><br>

Thor: Ragnarok	<br><br>

Harry Potter and the Sorcerer's Stone <br><br>

---

## Project Structure

```Movies-Recommendation-System/
|-- app.py                  # Main application file
|-- movie_list_update.pkl   # Preprocessed movie data
|-- preprocessing.ipynb     # Processing movie data file
|-- README.md               # Project documentation
|-- similarity.pkl          # Similarity matrix
|-- requirements.txt        # List of dependencies
```

---

## Future Improvements

- Collaborative Filtering: Incorporate user interaction data for better recommendations.<br><br>

- Hybrid Approach: Combine content-based and collaborative filtering for enhanced accuracy.<br><br>

- Improved UI: Add more features like search filters and movie descriptions. 🌟📈🎯<br><br>

## About Me

👋 Hi, I’m Pratham Gupta! I’m a Machine Learning enthusiast passionate about building data-driven applications. Feel free to check out my other projects on GitHub. 🚀🤖💡<br><br>

## Acknowledgments

The dataset used in this project is adapted from publicly available movie information.<br><br>

Special thanks to the creators of Streamlit and the Python ecosystem for making this project possible! 🎉🙏💻<br><br>

