# Crop Recommendation System

![alt text](download.jpeg)

This project is a machine learning-based Crop Recommendation System that predicts the most suitable crop to grow based on various environmental and soil parameters. The system uses a Random Forest Classifier trained on a dataset containing features such as Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.

## Features
- Exploratory Data Analysis (EDA) with visualizations for each feature
- Data preprocessing and scaling
- Model training and evaluation using Random Forest Classifier
- Model persistence with pickle
- Crop prediction for new input data
- Interactive web application for easy crop recommendations

## Dataset
The dataset used is `Crop_recommendation.csv`, which contains the following columns:
- N: Nitrogen content in soil
- P: Phosphorus content in soil
- K: Potassium content in soil
- temperature: Temperature in °C
- humidity: Relative humidity in %
- ph: pH value of the soil
- rainfall: Rainfall in mm
- label: Crop name

## Usage
1. **Install dependencies**
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - streamlit
   - pickle (standard library)

2. **Run the notebook**
   - Open `classification.ipynb` in VS Code or Jupyter Notebook.
   - Execute the cells sequentially to perform EDA, train the model, and make predictions.

3. **Model Prediction**
   - The notebook demonstrates how to use the trained model to predict the crop for new input values.

4. **Run the Streamlit App**
   - Execute `streamlit run app.py` to launch the interactive web application.
   - Use the sliders and input fields to enter soil and climate parameters.
   - Click the "Predict Crop" button to get a crop recommendation with detailed information.

## Files
- `classification.ipynb`: Main notebook containing code for EDA, model training, and prediction.
- `Crop_recommendation.csv`: Dataset file.
- `model.pkl`: Saved Random Forest model.
- `app.py`: Streamlit web application for interactive crop recommendations.

## Streamlit Application

The Streamlit app provides an intuitive user interface for the crop recommendation system:

- **Input Parameters**: Users can adjust soil nutrients (N, P, K, pH) and climate conditions (temperature, humidity, rainfall) using sliders and input fields.
- **Prediction**: The app uses the trained model to predict the most suitable crop based on the input parameters.
- **Crop Information**: Detailed information about the recommended crop is displayed, including ideal growing conditions.
- **User Guide**: Instructions for using the app are provided in the sidebar.

To run the application:
```
streamlit run app.py
```


