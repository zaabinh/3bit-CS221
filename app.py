import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ── Paths ────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
MODEL_DIR     = os.path.join(BASE_DIR, "models", "sbert")
ENCODER_DIR   = os.path.join(MODEL_DIR, "retrained_encoder")   # dùng retrained (F1 cao hơn)
CLASSIFIER_DIR= os.path.join(MODEL_DIR, "classifiers", "retrained")

# ── Load encoder + classifier + metadata ─────────────────────
try:
    encoder    = SentenceTransformer(ENCODER_DIR)
    classifier = joblib.load(os.path.join(CLASSIFIER_DIR, "model.joblib"))

    with open(os.path.join(CLASSIFIER_DIR, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)

    ASPECT_COLS         = metadata["aspect_cols"]
    THRESHOLDS          = metadata["thresholds"]
    NORMALIZE_EMBEDDINGS= metadata.get("normalize_embeddings", True)
    BATCH_SIZE          = metadata.get("batch_size", 32)

    print("✅ SBERT model loaded!")
    print("   Aspects   :", ASPECT_COLS)
    print("   Thresholds:", THRESHOLDS)

except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    encoder = classifier = metadata = None
    ASPECT_COLS = ["food", "service", "price", "ambiance", "miscellaneous"]
    THRESHOLDS  = {a: 0.5 for a in ASPECT_COLS}


# ── Predict ──────────────────────────────────────────────────
def predict(text: str):
    # 1. Encode text → embedding vector
    embedding = encoder.encode(
        [text],
        batch_size=BATCH_SIZE,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )

    # 2. Logistic Regression predict probabilities
    probs = classifier.predict_proba(embedding)   # shape (1, n_aspects)

    # 3. Apply per-aspect thresholds (từ metadata.json)
    predictions = np.zeros(len(ASPECT_COLS), dtype=np.int8)
    for i, aspect in enumerate(ASPECT_COLS):
        if probs[0, i] >= THRESHOLDS.get(aspect, 0.5):
            predictions[i] = 1

    # Đảm bảo ít nhất 1 aspect
    if predictions.sum() == 0:
        predictions[np.argmax(probs[0])] = 1

    aspects       = [ASPECT_COLS[i] for i, v in enumerate(predictions) if v == 1]
    probabilities = {asp: round(float(probs[0, i]), 3)
                     for i, asp in enumerate(ASPECT_COLS)}

    return {"aspects": aspects, "probabilities": probabilities}


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "Vui lòng nhập review."}), 400
    if encoder is None:
        return jsonify({"error": "Model chưa load được. Kiểm tra thư mục models/sbert/."}), 503

    try:
        return jsonify(predict(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)