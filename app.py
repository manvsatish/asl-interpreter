import streamlit as st
import cv2
import numpy as np
from gesture_recognition import predict_sign
from utils import speak

st.title("ASL Alphabet Translator")

img = st.camera_input("Take a picture")

if img is not None:
    # Convert image to OpenCV format
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    prediction, annotated_frame = predict_sign(frame)

    if prediction:
        st.success(f"Detected Sign: **{prediction}**")
        if st.button('Speak'):
            speak(prediction)

    # Show annotated image
    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Annotated Frame")