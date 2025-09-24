from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
model = data["model"]
le = data["le"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        N = float(request.form.get("N", 0))
        P = float(request.form.get("P", 0))
        K = float(request.form.get("K", 0))
        pH = float(request.form.get("pH", 6.5))
        rainfall = float(request.form.get("rainfall", 100))
        temperature = float(request.form.get("temperature", 25))

        features = np.array([[N, P, K, pH, rainfall, temperature]])
        pred = model.predict(features)[0]
        crop = le.inverse_transform([pred])[0]

        # Simple fertilizer suggestion heuristic (example)
        fert = []
        if N < 60:
            fert.append("Apply nitrogen-rich fertilizer (e.g., urea)")
        if P < 30:
            fert.append("Add phosphorus (e.g., single super phosphate)")
        if K < 40:
            fert.append("Add potash (e.g., muriate of potash)")

        return jsonify({
            "recommended_crop": crop,
            "fertilizer_suggestions": fert,
            "input": {"N":N,"P":P,"K":K,"pH":pH,"rainfall":rainfall,"temperature":temperature}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
