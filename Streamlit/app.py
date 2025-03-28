import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

# Streamlit Page Config
st.set_page_config(page_title="Facial Emotion Detector", page_icon="😊", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4A90E2;
        }
        .subheader {
            text-align: center;
            font-size: 20px;
            color: #333;
        }
        .prediction-box {
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #fff;
        }
        .angry { background-color: #E74C3C; }
        .disgust { background-color: #27AE60; }
        .fear { background-color: #8E44AD; }
        .happy { background-color: #F1C40F; }
        .neutral { background-color: #7F8C8D; }
        .sad { background-color: #3498DB; }
        .surprise { background-color: #E67E22; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<p class="title">😊 Facial Emotion Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload an image and let AI detect the emotion!</p>', unsafe_allow_html=True)

# Load Pretrained Model
@st.cache_resource
def load_emotion_model():
    model = load_model('best_model.keras')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_emotion_model()

# Emotion Labels
emotion_labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

# File Uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

# Preprocess Image
def preprocess_image(image_file, target_size=(48, 48)):
    img = Image.open(image_file).convert('L').resize(target_size)  # Convert to grayscale & resize
    img_array = img_to_array(img) / 255.0  # Normalize
    return np.reshape(img_array, (1, 48, 48, 1))

# Predict Emotion
if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🤖 Analyzing emotion..."):
        img_array = preprocess_image(uploaded_file)
        predictions = model.predict(img_array, verbose=0)
        predicted_label = np.argmax(predictions)
        confidence = np.max(predictions) * 100  # Confidence score

    # Display Prediction
    emotion = emotion_labels[predicted_label]
    st.markdown(
        f'<div class="prediction-box {emotion}">Predicted Emotion: {emotion.capitalize()} ({confidence:.2f}%)</div>',
        unsafe_allow_html=True
    )
