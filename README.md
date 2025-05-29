# Real-Time ASL Translator

## Overview
Translates American Sign Language (ASL) alphabet gestures into text and speech using computer vision and deep learning.

## Features
- Webcam-based real-time hand landmark detection (MediaPipe)
- Gesture classification with RandomForest model
- Streamlit app for easy use

## Installation
1. Create and activate a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Download ASL Alphabet dataset and place in `data/asl_alphabet_train`
4. Run training: `python train_model.py`
5. Run app: `streamlit run app.py`

## Usage
- Check "Start Webcam" to begin sign recognition.
- Detected signs show up as text.

## Future Improvements
- Support full ASL words/sentences with LSTM/Transformer models
- Increase dataset variety with user-collected data
