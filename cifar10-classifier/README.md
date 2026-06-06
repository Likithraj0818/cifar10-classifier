# 🔍 AI Product Classification System

A CNN-based image classifier trained on the CIFAR-10 dataset, deployed via a **Flask REST API** with a **Streamlit UI** for live inference.

## 🗂️ Project Structure

```
cifar10-classifier/
│
├── app.py                  # Flask REST API
├── streamlit_app.py        # Streamlit UI
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── notebooks/
│   ├── CNN_cifar_data.ipynb                     # CNN model training (CIFAR-10)
│   ├── Deepleraning_tensorflow_CNN_.ipynb        # TensorFlow/Keras deep learning
│   └── Chapter_2_OpenCV_Image_Opertations.ipynb # OpenCV image preprocessing
│
└── README.md
```

> ⚠️ `model.h5` is not included in this repo (too large). See **Setup** below.

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Model | CNN — TensorFlow / Keras |
| Image Processing | OpenCV |
| API | Flask |
| UI | Streamlit |
| Dataset | CIFAR-10 (10 classes, 60,000 images) |

---

## 📦 Classes

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/cifar10-classifier.git
cd cifar10-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your trained model
Train the model using the notebooks, then save it:
```python
model.save("model.h5")
```
Place `model.h5` in the project root.

### 4. Start the Flask API
```bash
python app.py
```
API runs at `http://localhost:5000`

### 5. Start the Streamlit UI
```bash
streamlit run streamlit_app.py
```
UI runs at `http://localhost:8501`

---

## 🔌 API Reference

### `GET /`
Health check.

### `POST /predict`
Upload an image and get a prediction.

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "predicted_class": "cat",
  "confidence": 87.34,
  "all_probabilities": {
    "airplane": 0.001,
    "automobile": 0.002,
    "bird": 0.012,
    "cat": 0.8734,
    ...
  }
}
```

---

## 🕷️ Scraping Pipeline

Automatically scrapes Google Images for all 10 CIFAR-10 categories.

```bash
# Scrape 50 images per class (default)
python scraper.py

# Scrape 200 images per class
python scraper.py --images 200

# Scrape only one class
python scraper.py --class airplane

# Clean existing dataset and re-scrape
python scraper.py --clean --images 100
```

Output structure:
```
dataset/
├── airplane/    (50 images)
├── automobile/  (50 images)
├── bird/        (50 images)
├── cat/         (50 images)
├── deer/        (50 images)
├── dog/         (50 images)
├── frog/        (50 images)
├── horse/       (50 images)
├── ship/        (50 images)
└── truck/       (50 images)
```
