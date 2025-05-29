import streamlit as st
import cv2
import time
from gesture_recognition import predict_sign

st.set_page_config(layout="wide")
st.title("🤟 Real-Time ASL Alphabet Translator")

FRAME_WINDOW = st.image([])
prediction_placeholder = st.empty()

start = st.checkbox("Start Webcam")

# Use AVFoundation backend on macOS
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if start:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access webcam")
            break

        prediction, annotated = predict_sign(frame)

        FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        if prediction:
            prediction_placeholder.markdown(f"### Detected Sign: **{prediction}**")
        else:
            prediction_placeholder.markdown("### No hand detected")

        time.sleep(0.1)

        if not st.session_state.get("Start Webcam", True):
            break
else:
    cap.release()
    prediction_placeholder.markdown("### Webcam Off")