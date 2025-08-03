import os
import cv2
import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__, static_folder='static')

# Load trained model
model_path = os.path.join('model', 'svm_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)

        prediction = model.predict(img)[0]
        confidence = model.predict_proba(img)[0][prediction]
        label = 'Cat 🐱' if prediction == 0 else 'Dog 🐶'

        return render_template('index.html', label=label, confidence=f"{confidence*100:.2f}%")
    except Exception as e:
        return render_template('index.html', label="Prediction failed", confidence=str(e))

if __name__ == '__main__':
    app.run(debug=True)