import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import joblib

def load_images_from_folder(folder_path, label, img_size=(64, 64)):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            data.append(img.flatten())
            labels.append(label)
    return data, labels

def train_and_save_model():
    cat_folder = os.path.join('data', 'train', 'cats')
    dog_folder = os.path.join('data', 'train', 'dogs')

    cat_data, cat_labels = load_images_from_folder(cat_folder, 0)
    dog_data, dog_labels = load_images_from_folder(dog_folder, 1)

    X = np.array(cat_data + dog_data)
    y = np.array(cat_labels + dog_labels)

    print(f"✅ Dataset loaded: {X.shape[0]} samples, each with {X.shape[1]} features")

    X, y = shuffle(X, y, random_state=42)
    X = X[:2000]
    y = y[:2000]
    print(f"⚡ Using reduced dataset: {X.shape[0]} samples")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', probability=True)
    print("⏳ Training model...")
    model.fit(X_train, y_train)

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, os.path.join('model', 'svm_model.pkl'))

    print(f"✅ Model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")

if __name__ == '__main__':
    train_and_save_model()