## 🐾 Dog vs Cat Recognition Web App

A full-stack machine learning web application that classifies uploaded images as either a *dog* or a *cat* using a trained SVM model. Built during my internship at *SkillCraft* as Task 3.

---

### 🚀 Features

- Upload an image and get instant prediction: *Dog 🐶 or Cat 🐱*
- Confidence score displayed for each prediction
- Clean and responsive UI with custom background
- Error handling for invalid inputs
- Fast training using sample reduction

---

### 🧠 Tech Stack

| Layer        | Tools Used                          |
|--------------|-------------------------------------|
| ML Model     | SVM (Scikit-learn)                  |
| Preprocessing| OpenCV, NumPy                       |
| Backend      | Flask                               |
| Frontend     | HTML, CSS                           |
| Deployment   | Localhost (Flask dev server)        |

---

### 📁 Folder Structure


project/
├── static/
│   ├── style.css
│   └── bg.png
├── templates/
│   └── index.html
├── model/
│   └── svm_model.pkl
├── data/
│   └── train/
│       ├── cats/
│       └── dogs/
├── app.py
├── train_svm.py
└── README.md


---

### ⚙ Setup Instructions

1. *Clone the repository*
   bash
   git clone https://github.com/yourusername/dog-cat-classifier.git
   cd dog-cat-classifier
   

2. *Install dependencies*
   bash
   pip install -r requirements.txt
   

3. *Prepare training data*
   - Place grayscale images of cats and dogs in data/train/cats/ and data/train/dogs/

4. *Train the model*
   bash
   python train_svm.py
   

5. *Run the web app*
   bash
   python app.py
   

6. *Open in browser*
   
   http://localhost:5000
   

---
