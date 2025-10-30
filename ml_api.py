# ml_api.py
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
# 1️⃣ Load trained model and label encoder
# -------------------------------------------------
model = None
label_encoder = None
feature_names = []

try:
    if not os.path.exists("disease_nn_model.h5") or not os.path.exists("label_encoder.pkl"):
        print("❌ Model or encoder file not found. Please run train_model.py first.")
        exit(1)

    model = tf.keras.models.load_model("disease_nn_model.h5")
    label_encoder = joblib.load("label_encoder.pkl")
    print("✅ Model and label encoder loaded successfully")

    # Load dataset just to get feature names
    df = pd.read_csv("DiseaseDataset.csv")
    feature_names = df.drop(columns=["diseases"]).columns.tolist()
    print(f"📋 Available features: {len(feature_names)} symptoms")

except Exception as e:
    print(f"❌ Error loading model or dataset: {e}")
    exit(1)

# -------------------------------------------------
# 2️⃣ Enable CORS (for Flutter)
# -------------------------------------------------
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
    return response

# -------------------------------------------------
# 3️⃣ Prediction Endpoint
# -------------------------------------------------
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204  # CORS preflight

    try:
        data = request.get_json()
        print(f"📨 Received data: {data}")

        # Prepare input data (initialize all to 0)
        input_data = {feature: 0 for feature in feature_names}

        # Map Flutter symptom selections to model features
        for symptom, is_selected in data.items():
            if symptom in input_data and is_selected:
                input_data[symptom] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict using Neural Network
        pred_probs = model.predict(input_df)
        pred_class = np.argmax(pred_probs)
        confidence = float(np.max(pred_probs))
        predicted_disease = label_encoder.inverse_transform([pred_class])[0]

        print(f"🎯 Prediction: {predicted_disease} (confidence: {confidence:.2f})")

        return jsonify({
            "prediction": predicted_disease,
            "confidence": confidence,
            "success": True
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e), 'success': False}), 400

# -------------------------------------------------
# 4️⃣ Health Check Endpoint
# -------------------------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "total_features": len(feature_names)
    })

# -------------------------------------------------
# 5️⃣ Features Endpoint (for Flutter symptom list)
# -------------------------------------------------
@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({
        "symptoms": feature_names
    })

# -------------------------------------------------
# 6️⃣ Run Flask Server
# -------------------------------------------------
if __name__ == '__main__':
    print("🚀 Starting Neural Network ML API server...")
    print("📊 Endpoints:")
    print("   - POST /predict")
    print("   - GET  /health")
    print("   - GET  /features")
    print("   - Running on http://0.0.0.0:5000")

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
