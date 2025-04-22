from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

# Set up the correct file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, 'Data', 'WineQT.csv')
model_file_path = os.path.join(current_dir, 'model', 'wine_quality_model.pkl')

# Load the Data
df = pd.read_csv(data_file_path)
# Check if 'quality' column exists before dropping
if 'quality' in df.columns:
    df = df.drop(columns=['quality'])
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Initialize model variable globally
model = None

# Load the model
try:
    model = pickle.load(open(model_file_path, 'rb'))
    print("Model loaded successfully!")
    print(f"Model expects features: {model.feature_names_in_}")
except Exception as e:
    print(f"Error loading model: {e}")
    
# Check if the model is loaded correctly
if model is None:
    print("WARNING: Model failed to load! The application may not work correctly.")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Error: Model not loaded. Please check the logs.')
    
    try:
        # Get form data
        form_values = request.form.to_dict()
        
        # Create feature array with correct feature names
        features = []
        features.append(float(form_values.get('fixed_acidity', 0)))
        features.append(float(form_values.get('volatile_acidity', 0)))
        features.append(float(form_values.get('citric_acid', 0)))
        features.append(float(form_values.get('residual_sugar', 0)))
        features.append(float(form_values.get('chlorides', 0)))
        features.append(float(form_values.get('free_sulfur_dioxide', 0)))
        features.append(float(form_values.get('total_sulfur_dioxide', 0)))
        features.append(float(form_values.get('density', 0)))
        features.append(float(form_values.get('pH', 0)))
        features.append(float(form_values.get('sulphates', 0)))
        features.append(float(form_values.get('alcohol', 0)))
        features.append(float(form_values.get('id', 1)))  # Added the ID feature
        
        # Convert features to numpy array
        final_features = np.array(features).reshape(1, -1)
        
        # Create DataFrame with correct feature names to ensure order
        input_df = pd.DataFrame([features], columns=model.feature_names_in_)
        
        # Make prediction using the DataFrame with proper column names
        prediction = model.predict(input_df)
        
        output = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text=f'Wine quality is: {output}')
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True, port=5000)