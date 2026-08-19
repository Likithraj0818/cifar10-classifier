# 📦 CIFAR-10 Image Classifier

> Upload an image, get an instant prediction across 10 object categories — powered by a CNN trained on CIFAR-10.

Built with **TensorFlow**, **Keras**, **OpenCV**, **Flask**, and **Streamlit**.

---

## 📌 What This Project Does

A convolutional neural network trained on the CIFAR-10 dataset (60,000 32×32 color images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The trained model is served two ways:

- A **Flask REST API** for programmatic predictions
- A **Streamlit UI** for interactive, real-time classification from uploaded images

**Example:**
```
Input:  upload a photo of a car
Output: "automobile" — 96.4% confidence
```

---

## 🏗️ Pipeline

```
Uploaded Image
     │
     ▼
OpenCV Preprocessing  ←── resize, normalize, reshape to (32, 32, 3)
     │
     ▼
CNN Model (TensorFlow/Keras)
     │
     ▼
Predicted Class + Confidence Score
     │
     ├──► Flask REST API response (JSON)
     └──► Streamlit UI (visual result)
```

---

## ⚙️ Tech Stack

- **TensorFlow / Keras** — CNN architecture, training, inference
- **OpenCV** — image preprocessing (resize, normalize, color conversion)
- **Flask** — REST API for serving predictions
- **Streamlit** — interactive web UI for live testing
- **NumPy** — array/tensor manipulation

---

## 🚀 How to Run

### 1. Clone the repo
```
git clone https://github.com/Likithraj0818/cifar10-classifier.git
cd cifar10-classifier
```

### 2. Install dependencies
```
pip install tensorflow opencv-python flask streamlit numpy
```

### 3. Run the Flask API
```
python app.py
```

### 4. Run the Streamlit UI
```
streamlit run streamlit_app.py
```

---

## 🔑 Key Concepts Implemented

- **CNN architecture** — convolutional + pooling layers for image feature extraction
- **Image preprocessing** — OpenCV-based resizing and normalization pipeline
- **Model serving** — both a REST API (Flask) and an interactive UI (Streamlit) for the same trained model
- **Real-time inference** — predictions returned on the fly from uploaded images

---

## 🧠 What I Learned

- Building and training a CNN from scratch for multi-class image classification
- Structuring an OpenCV preprocessing pipeline that matches training-time transformations exactly
- Serving the same model through two different interfaces (API vs. UI) without duplicating logic
- Trade-offs between batch inference (API) and interactive single-image inference (Streamlit)

---

## 👨‍💻 Author

**Nagula Likith Raj**

- 🔗 [LinkedIn](https://www.linkedin.com/in/likith-raj-b53794301/)
- 📧 nlikithraj9908@gmail.com
