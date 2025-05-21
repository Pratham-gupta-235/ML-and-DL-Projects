# Music Popularity Prediction

![alt text](Spotify-Earnings.webp)

A machine learning project that predicts the popularity of songs based on their audio features from Spotify data.

## Overview

This project uses machine learning techniques to analyze audio features of songs and predict their popularity scores. The model leverages various audio characteristics provided by Spotify's API to determine what makes a song popular.

## Dataset

The dataset ([`Spotify_data.csv`](Spotify_data.csv )) contains information about songs from Spotify, including:

- Track metadata (name, artist, album)
- Popularity scores (ranging from 0-100)
- Audio features:
  - Danceability
  - Energy
  - Loudness
  - Acousticness
  - Instrumentalness
  - Liveness
  - Valence
  - Tempo
  - Speechiness
  - Key
  - Mode
  - Duration

## Features Used

After correlation analysis and exploratory data analysis, the following audio features were identified as significant predictors of song popularity:

- Energy
- Valence
- Danceability
- Loudness
- Acousticness
- Tempo
- Speechiness
- Liveness

## Methodology

1. **Data Preprocessing**:
   - Loading and cleaning the dataset
   - Feature selection based on correlation with popularity
   - Data scaling

2. **Exploratory Data Analysis (EDA)**:
   - Correlation analysis between features
   - Distribution analysis of audio features
   - Visualizing relationships between features and popularity

3. **Model Development**:
   - Random Forest Regressor with hyperparameter tuning
   - GridSearchCV for finding optimal parameters

4. **Evaluation**:
   - Mean Squared Error (MSE)
   - R² Score
   - Actual vs. Predicted visualization

## Results

The Random Forest model was optimized through extensive hyperparameter tuning. Key findings include:
- Best parameters: max_depth=10, max_features='log2', min_samples_leaf=2, min_samples_split=5
- The model shows good predictive performance as visualized in the actual vs. predicted plot
- Certain audio features, such as energy and danceability, have stronger correlations with popularity

## Key Insights

- Higher energy levels and danceability tend to correlate positively with higher popularity scores
- Increased acousticness and lower loudness levels generally correspond with lower popularity
- The emotional positivity of a track (valence) has a weaker relationship with popularity

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Use

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the Jupyter notebook [`prediction.ipynb`](prediction.ipynb ) to see the analysis and model development process



