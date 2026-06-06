from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

MODEL_PATH = "model.h5"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    else:
        print(f"⚠️  Warning: '{MODEL_PATH}' not found. Place your trained model.h5 in the project root.")

def preprocess_image(file_bytes):
    """Preprocess uploaded image to CIFAR-10 format: 32x32 RGB, normalized."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # shape: (1, 32, 32, 3)
    return img

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "CIFAR-10 Classifier API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Place model.h5 in the project root."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send image with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img = preprocess_image(file.read())
        predictions = model.predict(img)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])
        all_probs = {CLASS_NAMES[i]: round(float(predictions[i]), 4) for i in range(10)}

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "all_probabilities": all_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
