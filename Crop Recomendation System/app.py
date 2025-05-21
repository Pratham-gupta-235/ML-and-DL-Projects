import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)


# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Main header
st.title('Crop Recommendation System🌾')

# Description
st.write("""
This application predicts the most suitable crop to grow based on the soil and climate conditions.
Enter the values for the parameters below and click the 'Predict' button to get a recommendation.
""")

# Create three columns for inputs with the middle one as a spacer
col1, space, col2 = st.columns([10, 1, 10])

with col1:
    st.subheader('Soil Nutrients')
    n = st.number_input('Nitrogen (N) content in soil', min_value=0, max_value=150, value=80)
    p = st.number_input('Phosphorus (P) content in soil', min_value=0, max_value=150, value=40)
    k = st.number_input('Potassium (K) content in soil', min_value=0, max_value=150, value=40)
    ph = st.slider('pH value of soil', min_value=0.0, max_value=14.0, value=6.5, step=0.01)

with col2:
    st.subheader('Climate Conditions')
    temperature = st.slider('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.slider('Humidity (%)', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    rainfall = st.slider('Rainfall (mm)', min_value=0.0, max_value=300.0, value=200.0, step=0.1)

# Create a prediction button
predict_button = st.button("Predict Crop", use_container_width=True)

# Display crop description based on prediction
def display_crop_info(crop_name):
    crop_descriptions = {
        'rice': "Rice thrives in warm, humid conditions with high rainfall. It requires waterlogged soil and high nitrogen content.",
        'maize': "Maize (corn) grows well in well-drained soils with moderate nitrogen and warm temperatures.",
        'chickpea': "Chickpeas prefer moderate temperatures and low humidity, with well-drained soil.",
        'kidneybeans': "Kidney beans need warm temperatures, moderate rainfall, and soil with good drainage.",
        'pigeonpeas': "Pigeon peas are drought-resistant and prefer warm temperatures with well-drained soil.",
        'mothbeans': "Moth beans are highly drought-resistant and grow well in arid conditions with minimal rainfall.",
        'mungbean': "Mung beans prefer warm temperatures and moderate rainfall with well-drained soil.",
        'blackgram': "Black gram thrives in warm, humid conditions with moderate rainfall.",
        'lentil': "Lentils grow best in cool temperatures and need well-drained soil with neutral pH.",
        'pomegranate': "Pomegranates prefer hot, dry climates and thrive in well-drained soil with moderate fertility.",
        'banana': "Bananas need tropical conditions with high humidity, rainfall, and nitrogen-rich soil.",
        'mango': "Mangoes thrive in tropical climates with a distinct dry season and moderate soil fertility.",
        'grapes': "Grapes require warm temperatures, low rainfall during ripening, and well-drained soil.",
        'watermelon': "Watermelons need warm temperatures, moderate water, and well-drained, sandy soil.",
        'muskmelon': "Muskmelons prefer warm, dry conditions and well-drained soil rich in organic matter.",
        'apple': "Apples need cool temperatures, moderate rainfall, and well-drained soil with neutral pH.",
        'orange': "Oranges thrive in subtropical climates with moderate rainfall and well-drained soil.",
        'papaya': "Papayas need tropical conditions with high humidity, rainfall, and rich, well-drained soil.",
        'coconut': "Coconuts require tropical coastal conditions with high humidity and sandy, well-drained soil.",
        'cotton': "Cotton grows best in warm climates with moderate rainfall and well-drained soil.",
        'jute': "Jute requires warm, humid conditions with high rainfall and loamy soil rich in organic matter.",
        'coffee': "Coffee thrives in subtropical climates with moderate temperatures, rainfall, and well-drained soil."
    }
    
    if crop_name.lower() in crop_descriptions:
        st.info(crop_descriptions[crop_name.lower()])
    else:
        st.info("No additional information available for this crop.")

# Make a prediction when the button is clicked
if predict_button:
    try:
        # Load the model
        model = load_model()
        
        # Create a feature array
        features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Display the result
        st.header(f'Recommended Crop: {prediction[0]}')
        
        # Display additional information about the crop
        st.subheader("Crop Information:")
        display_crop_info(prediction[0])
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
# Add information to sidebar
with st.sidebar:
    st.markdown("# About Me")
    intro_text = """
    👋 Hi there!  
    I'm **Pratham Gupta**, and I'm thrilled to share my latest **Machine Learning** project: the **Crop Recommendation System**.  
    This system uses a **Random Forest Classifier** to provide personalized crop suggestions based on soil and climate conditions.  

    ### Curious About the Code?  
    The full implementation is available on my GitHub, complete with documentation to help you dive right in.  
    👉 [Visit My GitHub](https://github.com/Code1235/Machine-Learning-Projects/tree/main/Crop%20Recomendation%20System)  

    Thanks for stopping by, and happy exploring!
    """
    st.markdown(intro_text)
    
    st.markdown("---")
    
    st.markdown("### How to use this app")
    st.markdown("""
    1. Enter the values for soil nutrients (N, P, K) and pH.
    2. Set the climate conditions (temperature, humidity, rainfall).
    3. Click the 'Predict Crop' button to get a recommendation.
    4. The app will suggest the most suitable crop based on your inputs.
    """)
    
    st.markdown("---")
    
    st.markdown("### About this project")
    st.markdown("""
    This Crop Recommendation System uses a Random Forest Classifier machine learning model trained on a dataset 
    containing soil parameters and climate conditions for various crops. The model predicts the most suitable 
    crop based on the input features.
    
    **Features used for prediction:**
    - Nitrogen (N): Nitrogen content in soil
    - Phosphorus (P): Phosphorus content in soil
    - Potassium (K): Potassium content in soil
    - Temperature: Temperature in degrees Celsius
    - Humidity: Relative humidity in percentage
    - pH: pH value of the soil
    - Rainfall: Rainfall in millimeters
    """)