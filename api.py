"""
api.py - Flask API serving four ML artifacts:
regression (SME revenue), classification (SME credit risk),
clustering (mall customer segment), association rules (recommender).
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",
    "null",  # allows opening a frontend .html file directly (file://)
]}})

# Load all artifacts ONCE at startup (never retrain or reload per request)
MODEL_DIR = "./model"

regression_pipeline = joblib.load(os.path.join(MODEL_DIR, "lasso_regressor_for_sme_revenue.joblib"))
classification_pipeline = joblib.load(os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib"))

clustering_artifact = joblib.load(os.path.join(MODEL_DIR, "kmeans_mall_customer_segmentation.joblib"))
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

arm_artifact = joblib.load(os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib"))
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

print("All the four artifacts have been loaded. The API is ready.")


def get_payload():
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        raise ValueError("Request body must be a non-empty JSON object")
    return payload


# 1. REGRESSION
@app.route("/api/v1/models/sme-revenue-regressor/predictions", methods=["POST"])
def predict_revenue():
    input_row = pd.DataFrame([get_payload()])
    predicted_log_revenue = regression_pipeline.predict(input_row)[0]
    # Model was trained on log1p(revenue); invert it to report KES
    predicted_revenue_kes = float(np.expm1(predicted_log_revenue))
    return jsonify({
        "predicted_monthly_revenue_kes": round(predicted_revenue_kes, 2),
        "model": "Lasso Regressor",
    })


# 2. CLASSIFICATION
@app.route("/api/v1/models/sme-credit-risk-classifier/predictions", methods=["POST"])
def predict_credit_risk():
    input_row = pd.DataFrame([get_payload()])
    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_
    return jsonify({
        "predicted_credit_risk_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Support Vector Classifier (tuned, SMOTE-resampled at training only)",
    })


# 3. CLUSTERING
@app.route("/api/v1/models/mall-customer-segmenter/predictions", methods=["POST"])
def predict_customer_segment():
    input_row = pd.DataFrame([get_payload()])
    scaled_input = clustering_scaler.transform(input_row)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])
    return jsonify({
        "assigned_cluster": assigned_cluster,
        "model": "K-Means (k=5, standardized features)",
    })


# 4. ASSOCIATION RULES
@app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
def recommend_products():
    payload = get_payload()
    cart = set(payload.get("cart", []))
    if not cart:
        raise ValueError("'cart' must be a non-empty list of SKUs")
    top_n = int(payload.get("top_n", 3))

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "sku": str(sku),
            "product_name": sku_lookup.loc[sku, "product_name"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for sku, (confidence, lift) in ranked
    ]
    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "model": "Apriori association rules, cleaned + precomputed lookup",
    })


# Error handling: clean JSON instead of HTML error pages
@app.errorhandler(ValueError)
@app.errorhandler(KeyError)
def handle_bad_input(e):
    return jsonify({"error": f"Invalid or missing input: {e}", "status": "failed"}), 400


@app.errorhandler(Exception)
def handle_server_error(e):
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description, "status": "failed"}), e.code
    return jsonify({"error": f"Inference error: {e}", "status": "failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)