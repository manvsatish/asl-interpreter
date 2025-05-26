import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y])
        return landmarks
    return None

def load_dataset(dataset_path):
    X, y = [], []
    labels = [label for label in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, label))]

    for label in tqdm(labels, desc="Classes"):
        label_path = os.path.join(dataset_path, label)
        images = os.listdir(label_path)

        for img_file in tqdm(images, desc=f"Processing {label}", leave=False):
            img_path = os.path.join(label_path, img_file)
            landmarks = extract_landmarks(img_path)
            if landmarks:
                X.append(landmarks)
                y.append(label)
    return np.array(X), np.array(y)

def main():
    dataset_path = "data/asl_alphabet_train"
    print("Loading dataset...")
    X, y = load_dataset(dataset_path)
    print(f"Loaded {len(X)} samples.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100)
    print("Training model...")
    clf.fit(X_train, y_train)

    print(f"Training accuracy: {clf.score(X_train, y_train):.2f}")
    print(f"Test accuracy: {clf.score(X_test, y_test):.2f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/asl_classifier.pkl")
    print("Model saved to model/asl_classifier.pkl")

if __name__ == "__main__":
    main()