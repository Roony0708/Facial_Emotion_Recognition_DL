import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.models import load_model
import cv2

# Load the pretrained model
pretrained_model = load_model('best_model.keras')

# Define the emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Initialize video capture
video = cv2.VideoCapture(0)

# Load the Haar cascade for face detection
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit button to start/stop the camera
st.title("Real-Time Face Emotion Detection")
start_button = st.button("Start Camera")

if start_button:
    while True:
        ret, frame = video.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        for x, y, w, h in faces:
            sub_face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = pretrained_model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, emotion_labels[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Real-Time Emotion Detection", frame)

        # Stop the loop if 'q' is pressed or window is closed
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Real-Time Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

# Release resources when exiting
video.release()
cv2.destroyAllWindows()
