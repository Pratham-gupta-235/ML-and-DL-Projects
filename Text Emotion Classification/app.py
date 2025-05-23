import pickle
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import pad_sequences, SimpleTokenizer, emoji_map, color_map

# Set page config
st.set_page_config(
    page_title="Text Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .emotion-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        background-color: black;
    }
    .emoji-large {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    .title {
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        saved_objects = pickle.load(open('text_classification.pkl', 'rb'))
        
        # Extract model, tokenizer, and other saved objects
        if isinstance(saved_objects, dict):
            model = saved_objects.get('model')
            tokenizer = saved_objects.get('tokenizer')
            max_length = saved_objects.get('max_length')
            encoder = saved_objects.get('encoder')
        else:
            model = saved_objects
            
            # Load the data to recreate tokenizer and encoder
            data = pd.read_csv('Dataset/train.txt', sep=';')
            data.columns = ['Text', 'Emotions']
            
            # Recreate tokenizer
            texts = data['Text'].tolist()
            tokenizer = SimpleTokenizer()
            tokenizer.fit_on_texts(texts)
            
            # Recreate sequences and padding
            sequences = tokenizer.texts_to_sequences(texts)
            max_length = max([len(seq) for seq in sequences])
            
            # Recreate label encoder
            encoder = LabelEncoder()
            encoder.fit_transform(data['Emotions'].tolist())
            
        return model, tokenizer, max_length, encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def predict_emotion(input_text, model, tokenizer, max_length, encoder):
    # Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequences = pad_sequences(input_sequence, maxlen=max_length)
    
    # Make prediction
    prediction = model.predict(padded_input_sequences)[0]
    
    # Get the predicted emotion
    predicted_index = np.argmax(prediction)
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    emoji = emoji_map.get(predicted_label, '')
    
    # Prepare confidence scores
    confidence_scores = []
    for i, prob in enumerate(prediction):
        label = encoder.inverse_transform([i])[0]
        confidence_scores.append({
            'label': label,
            'probability': float(prob * 100),
            'emoji': emoji_map.get(label, ''),
            'color': color_map.get(label, '#007bff')
        })
    
    # Sort by probability
    confidence_scores.sort(key=lambda x: x['probability'], reverse=True)
    
    return predicted_label, emoji, float(prediction[predicted_index] * 100), confidence_scores

def main():
    # Load model and dependencies
    model, tokenizer, max_length, encoder = load_model()
    
    # App title
    st.markdown("<h1 class='title'>Text Emotion Classifier</h1>", unsafe_allow_html=True)
    
    # Text input
    input_text = st.text_area("Enter text to analyze:", 
                              placeholder="Type your text here... (e.g., 'I'm so excited about the party tonight!')",
                              height=150)
    
    # Analyze button
    if st.button("Analyze Emotion"):
        if not input_text:
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                predicted_label, emoji, confidence, confidence_scores = predict_emotion(
                    input_text, model, tokenizer, max_length, encoder
                )
            
            # Display the result in a card
            st.markdown(f"""
            <div class="emotion-card">
                <div class="emoji-large">{emoji}</div>
                <h2>{predicted_label.capitalize()}</h2>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display confidence scores with progress bars
            st.markdown("### All Emotions:")
            for score in confidence_scores:
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"<h4>{score['emoji']} {score['label']}</h4>", unsafe_allow_html=True)
                with col2:
                    st.progress(score['probability'] / 100)
                    st.text(f"{score['probability']:.2f}%")
                st.markdown("---")
    
    # Footer
    st.markdown("<div class='footer'>Text Emotion Classification Model © 2025</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
