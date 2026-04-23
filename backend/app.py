import random
import io
import csv
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_wqi, FEATURES
from recommender import get_recommendations

app = Flask(__name__)
CORS(app)   # allow the frontend (different port) to call these APIs


# ── health check ────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "AquaVision API is running"})


# ── predict WQI from JSON body ───────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    result = predict_wqi(data)
    result["recommendations"] = get_recommendations(data)
    result["input"] = data
    return jsonify(result)


# ── upload CSV and predict for every row ─────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    content = file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)

    if not rows:
        return jsonify({"error": "CSV is empty"}), 400

    results = []
    for i, row in enumerate(rows):
        try:
            input_data = {f: float(row.get(f, 0)) for f in FEATURES}
            pred = predict_wqi(input_data)
            pred["row"] = i + 1
            pred["input"] = input_data
            # include lat/lng if present in CSV
            if "latitude" in row:
                pred["latitude"] = float(row["latitude"])
            if "longitude" in row:
                pred["longitude"] = float(row["longitude"])
            results.append(pred)
        except (ValueError, KeyError):
            results.append({"row": i + 1, "error": "Invalid or missing values"})

    return jsonify({"total_rows": len(results), "results": results})


# ── feature importance (standalone endpoint) ─────────────────────────────────
@app.route("/api/features", methods=["POST"])
def features():
    data = request.get_json() or {}
    result = predict_wqi(data)
    return jsonify({
        "feature_importance": result["feature_importance"],
        "model_used": result["model_used"],
    })


# ── simulated real-time sensor reading ───────────────────────────────────────
@app.route("/api/stream")
def stream():
    reading = {
        "ph":               round(random.uniform(6.0, 9.0), 2),
        "dissolved_oxygen": round(random.uniform(2.0, 12.0), 2),
        "turbidity":        round(random.uniform(0.1, 20.0), 2),
        "conductivity":     round(random.uniform(50, 1200), 1),
        "bod":              round(random.uniform(0.5, 10.0), 2),
        "nitrates":         round(random.uniform(0.1, 20.0), 2),
        "total_coliform":   round(random.uniform(0, 5), 1),
    }
    result = predict_wqi(reading)
    result["reading"] = reading
    result["recommendations"] = get_recommendations(reading)
    return jsonify(result)

# ── trigger model training ────────────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def train_model():
    from model import train
    try:
        result = train()
        return jsonify({"status": "success", "results": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── SHAP values for a single input ───────────────────────────────────────────
@app.route("/api/shap", methods=["POST"])
def shap_values():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    from model import get_shap_values
    return jsonify(get_shap_values(data))
# ── standalone recommendations ────────────────────────────────────────────────
@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    return jsonify(get_recommendations(data))

if __name__ == "__main__":
    app.run(debug=True, port=5000)