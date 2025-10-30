
# ===============================================
# 🚀 Neural Network API for Disease Prediction
# ===============================================

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# -------------------------------------------------
# 1️⃣ Load model, encoder, and feature names
# -------------------------------------------------
model = None
label_encoder = None
feature_names = []

try:
    MODEL_PATH = "disease_nn_model.h5"
    ENCODER_PATH = "label_encoder.pkl"
    HEADER_PATH = "header.txt"  # Only first line from dataset

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("❌ Model or encoder file missing. Train model first.")

    # Load model and label encoder
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ Model and encoder loaded successfully")

    # Load only header row (symptom names)
    if os.path.exists(HEADER_PATH):
        with open(HEADER_PATH, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
            columns = [col.strip() for col in header_line.split(",")]
            feature_names = [c for c in columns if c.lower() != "diseases"]
        print(f"📋 Loaded {len(feature_names)} features from header.txt")
    else:
        raise FileNotFoundError("❌ header.txt not found.")

except Exception as e:
    print(f"❌ Initialization error: {e}")

# -------------------------------------------------
# 2️⃣ CORS
# -------------------------------------------------
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
    return response

# -------------------------------------------------
# 3️⃣ Prediction
# -------------------------------------------------
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        print(f"📨 Received: {data}")

        input_data = {f: 0 for f in feature_names}

        for symptom, selected in data.items():
            if symptom in input_data and selected:
                input_data[symptom] = 1

        input_df = pd.DataFrame([input_data])
        pred_probs = model.predict(input_df)
        pred_class = np.argmax(pred_probs)
        confidence = float(np.max(pred_probs))
        predicted_disease = label_encoder.inverse_transform([pred_class])[0]

        print(f"🎯 Predicted: {predicted_disease} ({confidence:.2f})")

        return jsonify({
            "prediction": predicted_disease,
            "confidence": confidence,
            "success": True
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e), 'success': False}), 400

# -------------------------------------------------
# 4️⃣ Health Check
# -------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "total_features": len(feature_names)
    })

# -------------------------------------------------
# 5️⃣ Features Endpoint
# -------------------------------------------------
@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({"symptoms": feature_names})

# -------------------------------------------------
# 6️⃣ Entry Point
# -------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("🚀 API running on port", port)
    app.run(host='0.0.0.0', port=port)
