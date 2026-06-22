from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ─────────────────────────────────────────────
def predict(text: str) -> list[str]:
    import joblib
    import numpy as np

    vectorizer  = joblib.load("models/tfidf_vectorizer.pkl")
    model       = joblib.load("models/tfidf_logistic_regression_model.pkl")
    thresholds  = joblib.load("models/tfidf_thresholds.pkl")
    aspect_cols = joblib.load("models/aspect_cols.pkl")

    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]

    aspects = []
    for i, aspect in enumerate(aspect_cols):
        if probs[i] >= thresholds.get(aspect, 0.5):
            aspects.append(aspect)

    return aspects if aspects else [aspect_cols[int(np.argmax(probs))]]
# ─────────────────────────────────────────────
def predict(text: str) -> list[str]:
    """
    Hàm giả lập model NLP.
    Trả về danh sách các aspect được phát hiện trong review.

    Ví dụ tích hợp model thật:
        from your_model import YourModel
        model = YourModel.load("path/to/weights")

        def predict(text):
            return model.predict(text)
    """
    text_lower = text.lower()
    aspects = []

    keyword_map = {
        "Food":         ["đồ ăn", "món", "thức ăn", "ngon", "dở", "tươi", "food", "meal", "dish", "taste"],
        "Service":      ["nhân viên", "phục vụ", "service", "staff", "waiter", "chậm", "nhanh", "thái độ"],
        "Prices":       ["giá", "rẻ", "đắt", "tiền", "price", "cost", "cheap", "expensive", "value"],
        "Ambience":     ["không khí", "decor", "ambience", "atmosphere", "nhạc", "yên tĩnh", "ồn", "đẹp", "cozy"],
        "Miscellaneous":["vệ sinh", "vị trí", "parking", "bãi xe", "wifi", "misc", "overall", "chung"],
    }

    for aspect, keywords in keyword_map.items():
        if any(kw in text_lower for kw in keywords):
            aspects.append(aspect)

    return aspects if aspects else ["Miscellaneous"]
# ─────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "Vui lòng nhập nội dung review."}), 400

    aspects = predict(text)
    return jsonify({"aspects": aspects})


if __name__ == "__main__":
    app.run(debug=True)
