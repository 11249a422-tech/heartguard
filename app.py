"""
HeartGuard - Heart Disease Risk Predictor
Flask Web Application & API
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "heartguard-clinical-secret-key-2026")

MEDICAL_DISCLAIMER = (
    "This is an educational project, not a medical diagnosis. "
    "Please consult a doctor."
)

# Global variables for model and metrics
model_pipeline = None
metrics_data = {}


def load_artifacts():
    """Load model pipeline and metrics json safely."""
    global model_pipeline, metrics_data
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
        except Exception as e:
            app.logger.error(f"Error loading model: {e}")
            model_pipeline = None

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception as e:
            app.logger.error(f"Error loading metrics: {e}")
            metrics_data = {}


load_artifacts()

# Feature metadata and descriptions
FEATURE_METADATA = {
    "age": {"name": "Age", "unit": "years", "min": 18, "max": 120, "type": "numeric"},
    "sex": {"name": "Biological Sex", "options": {1: "Male", 0: "Female"}, "type": "categorical"},
    "cp": {
        "name": "Chest Pain Type",
        "options": {
            1: "Typical Angina (discomfort with exertion)",
            2: "Atypical Angina (non-classical chest discomfort)",
            3: "Non-Anginal Pain (not related to heart ischemia)",
            4: "Asymptomatic (no chest pain symptoms)",
        },
        "type": "categorical",
    },
    "trestbps": {"name": "Resting Blood Pressure", "unit": "mm Hg", "min": 80, "max": 240, "type": "numeric"},
    "chol": {"name": "Serum Cholesterol", "unit": "mg/dl", "min": 100, "max": 600, "type": "numeric"},
    "fbs": {
        "name": "Fasting Blood Sugar > 120 mg/dl",
        "options": {0: "No (<= 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
        "type": "categorical",
    },
    "restecg": {
        "name": "Resting ECG",
        "options": {
            0: "Normal",
            1: "ST-T Wave Abnormality (T-inversion / ST depression)",
            2: "Left Ventricular Hypertrophy (Estes' criteria)",
        },
        "type": "categorical",
    },
    "thalach": {"name": "Maximum Heart Rate Achieved", "unit": "bpm", "min": 60, "max": 220, "type": "numeric"},
    "exang": {
        "name": "Exercise-Induced Angina",
        "options": {0: "No", 1: "Yes (angina precipitated by exercise)"},
        "type": "categorical",
    },
    "oldpeak": {"name": "ST Depression (Exercise vs Rest)", "unit": "mm", "min": 0.0, "max": 10.0, "type": "numeric"},
    "slope": {
        "name": "Peak Exercise ST Slope",
        "options": {
            1: "Upsloping (normal or mild exertion response)",
            2: "Flat (ischemic indicator)",
            3: "Downsloping (severe ischemia indicator)",
        },
        "type": "categorical",
    },
    "ca": {
        "name": "Major Vessels Colored by Fluoroscopy",
        "options": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
        "type": "categorical",
    },
    "thal": {
        "name": "Thalassemia Scintigraphy",
        "options": {
            3: "Normal Blood Flow",
            6: "Fixed Defect (prior infarction / scar tissue)",
            7: "Reversible Defect (stress ischemia)",
        },
        "type": "categorical",
    },
}


def validate_and_parse_input(data):
    """
    Validate input dictionary and convert values to standard numeric types.
    Returns (cleaned_dict, error_list).
    """
    cleaned = {}
    errors = []

    required_keys = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    for key in required_keys:
        if key not in data or data[key] is None or str(data[key]).strip() == "":
            errors.append(f"Missing field: '{key}' ({FEATURE_METADATA.get(key, {}).get('name', key)}) is required.")
            continue

        raw_val = data[key]
        meta = FEATURE_METADATA[key]

        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            errors.append(f"Invalid numeric value for '{key}': {raw_val}")
            continue

        # Range or option validation
        if meta["type"] == "numeric":
            if val < meta["min"] or val > meta["max"]:
                errors.append(
                    f"'{meta['name']}' value {val} is outside valid physiological range ({meta['min']} - {meta['max']} {meta.get('unit', '')})."
                )
            cleaned[key] = float(val)
        elif meta["type"] == "categorical":
            int_val = int(round(val))
            if int_val not in meta["options"]:
                valid_opts = list(meta["options"].keys())
                errors.append(f"'{meta['name']}' must be one of {valid_opts}, got {int_val}.")
            cleaned[key] = float(int_val)

    return cleaned, errors


def get_risk_tier(probability):
    """Return risk category badge, color and description."""
    pct = round(probability * 100, 1)
    if probability < 0.30:
        return {
            "level": "Low",
            "badge_class": "badge-low",
            "color": "#10b981",
            "hex_accent": "#059669",
            "description": "Your estimated risk is lower than the typical clinical baseline in this model. Continue heart-healthy habits.",
        }
    elif probability < 0.70:
        return {
            "level": "Moderate",
            "badge_class": "badge-moderate",
            "color": "#f59e0b",
            "hex_accent": "#d97706",
            "description": "Several clinical parameters elevate your cardiovascular risk profile. Diagnostic follow-up is recommended.",
        }
    else:
        return {
            "level": "High",
            "badge_class": "badge-high",
            "color": "#ef4444",
            "hex_accent": "#dc2626",
            "description": "Significant risk factors detected across key physiological markers. Prompt clinical evaluation is strongly advised.",
        }


def explain_prediction(patient_dict, pipeline):
    """
    Extract top factors influencing prediction using model weights and patient values.
    Returns list of dicts with name, impact ('increases_risk' or 'decreases_risk'), weight, explanation.
    """
    if pipeline is None:
        return []

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        df_input = pd.DataFrame([patient_dict])
        transformed_arr = preprocessor.transform(df_input)[0]

        num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        all_features = num_cols + cat_feature_names

        # Extract model coefficients or feature importances
        if hasattr(classifier, "coef_"):
            coefs = classifier.coef_[0]
            contributions = transformed_arr * coefs
        elif hasattr(classifier, "feature_importances_"):
            # Tree based weighting
            contributions = transformed_arr * classifier.feature_importances_
        else:
            return []

        factor_impacts = []
        for feat_name, impact in zip(all_features, contributions):
            if abs(impact) > 0.05:
                factor_impacts.append((feat_name, float(impact)))

        # Sort by absolute impact
        factor_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        explanations = []
        for feat_code, impact in factor_impacts[:6]:
            direction = "increases_risk" if impact > 0 else "decreases_risk"
            readable_desc = interpret_feature_code(feat_code, patient_dict, impact > 0)
            explanations.append({
                "code": feat_code,
                "impact": direction,
                "score": round(abs(impact), 2),
                "is_risk": impact > 0,
                "text": readable_desc,
            })

        return explanations
    except Exception as e:
        app.logger.warning(f"Error computing feature explanations: {e}")
        return []


def interpret_feature_code(code, patient_data, increases_risk):
    """Helper to convert feature code into patient-friendly clinical wording."""
    if code == "age":
        age_val = int(patient_data.get("age", 0))
        return f"Age ({age_val} yrs): {'Elevated age-related vascular risk' if increases_risk else 'Younger age profile protecting cardiovascular elasticity'}"
    if code == "trestbps":
        bp_val = int(patient_data.get("trestbps", 0))
        return f"Resting Blood Pressure ({bp_val} mm Hg): {'Elevated arterial pressure' if increases_risk else 'Healthy optimal resting pressure'}"
    if code == "chol":
        chol_val = int(patient_data.get("chol", 0))
        return f"Serum Cholesterol ({chol_val} mg/dl): {'Elevated circulating lipid levels' if increases_risk else 'Favorable total lipid profile'}"
    if code == "thalach":
        hr_val = int(patient_data.get("thalach", 0))
        return f"Maximum Heart Rate ({hr_val} bpm): {'Suboptimal chronotropic reserve during peak stress' if increases_risk else 'Robust exercise heart rate response'}"
    if code == "oldpeak":
        st_val = float(patient_data.get("oldpeak", 0))
        return f"ST Depression ({st_val} mm): {'Significant exercise-induced electrocardiographic ischemia' if increases_risk else 'Minimal ST segment deviation at rest'}"
    if code.startswith("cp_"):
        cp_type = code.split("_")[1].replace(".0", "")
        names = {"1": "Typical Angina", "2": "Atypical Angina", "3": "Non-Anginal Discomfort", "4": "Asymptomatic (Silent)"}
        cp_name = names.get(cp_type, f"Type {cp_type}")
        return f"Chest Pain ({cp_name}): {'Clinical marker correlating with ischemic presentations' if increases_risk else 'Atypical / lower correlation with obstructive disease'}"
    if code.startswith("ca_"):
        vessels = code.split("_")[1].replace(".0", "")
        return f"Coronary Fluoroscopy ({vessels} major vessels): {'Fluoroscopic evidence of arterial vessel involvement' if increases_risk else 'Absence of vessel narrowing on fluoroscopy'}"
    if code.startswith("thal_"):
        thal_type = code.split("_")[1].replace(".0", "")
        names = {"3": "Normal Perfusion", "6": "Fixed Defect", "7": "Reversible Defect"}
        t_name = names.get(thal_type, f"Thal {thal_type}")
        return f"Thallium Scintigraphy ({t_name}): {'Myocardial perfusion defect detected under stress' if increases_risk else 'Normal myocardial blood flow'}"
    if code.startswith("slope_"):
        slope_type = code.split("_")[1].replace(".0", "")
        names = {"1": "Upsloping", "2": "Flat", "3": "Downsloping"}
        s_name = names.get(slope_type, f"Slope {slope_type}")
        return f"Peak Exercise ST Slope ({s_name}): {'Pathological ECG response indicating ischemia' if increases_risk else 'Normal ascending exercise repolarization'}"
    if code.startswith("exang_"):
        val = code.split("_")[1].replace(".0", "")
        return f"Exercise Angina ({'Present' if val == '1' else 'Absent'}): {'Direct symptom of coronary blood flow shortfall under exertion' if increases_risk else 'Free of exertional angina'}"
    if code.startswith("sex_"):
        val = code.split("_")[1].replace(".0", "")
        return f"Biological Sex ({'Male' if val == '1' else 'Female'}): {'Epidemiologically higher baseline cohort risk' if increases_risk else 'Lower baseline statistical risk in study cohort'}"

    return f"{code}: {'Contributed positively to risk probability' if increases_risk else 'Contributed to lower risk probability'}"


# ==============================================================================
# Routes
# ==============================================================================


@app.route("/")
def index():
    """Home landing page explaining HeartGuard with call to action."""
    return render_template(
        "index.html",
        metrics=metrics_data,
        disclaimer=MEDICAL_DISCLAIMER,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Health details form (GET) and prediction processor (POST)."""
    if request.method == "GET":
        return render_template(
            "predict.html",
            metadata=FEATURE_METADATA,
            disclaimer=MEDICAL_DISCLAIMER,
            errors=[],
            form_data={},
        )

    # POST request from HTML form
    cleaned_data, errors = validate_and_parse_input(request.form.to_dict())

    if errors:
        return render_template(
            "predict.html",
            metadata=FEATURE_METADATA,
            disclaimer=MEDICAL_DISCLAIMER,
            errors=errors,
            form_data=request.form.to_dict(),
        ), 400

    if model_pipeline is None:
        load_artifacts()
        if model_pipeline is None:
            return render_template(
                "predict.html",
                metadata=FEATURE_METADATA,
                disclaimer=MEDICAL_DISCLAIMER,
                errors=["Trained model is not loaded. Please ensure model/heart_model.pkl exists."],
                form_data=request.form.to_dict(),
            ), 500

    # Model inference
    df_single = pd.DataFrame([cleaned_data])
    prob = float(model_pipeline.predict_proba(df_single)[0][1])
    pred = int(model_pipeline.predict(df_single)[0])
    risk_tier = get_risk_tier(prob)
    top_factors = explain_prediction(cleaned_data, model_pipeline)

    return render_template(
        "result.html",
        patient=cleaned_data,
        metadata=FEATURE_METADATA,
        probability=round(prob, 4),
        percentage=round(prob * 100, 1),
        prediction=pred,
        risk_tier=risk_tier,
        top_factors=top_factors,
        disclaimer=MEDICAL_DISCLAIMER,
        best_model=metrics_data.get("best_model", "Logistic Regression"),
    )


@app.route("/about")
def about():
    """Model performance, metrics, dataset details and clinical background."""
    return render_template(
        "about.html",
        metrics=metrics_data,
        metadata=FEATURE_METADATA,
        disclaimer=MEDICAL_DISCLAIMER,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint returning:
    {"risk_probability": ..., "risk_percentage": ..., "risk_level": ..., ...}
    """
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Content-Type must be application/json.",
            "disclaimer": MEDICAL_DISCLAIMER,
        }), 400

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({
            "status": "error",
            "message": "Invalid or empty JSON payload.",
            "disclaimer": MEDICAL_DISCLAIMER,
        }), 400

    cleaned_data, errors = validate_and_parse_input(payload)
    if errors:
        return jsonify({
            "status": "error",
            "message": "Input validation failed.",
            "errors": errors,
            "disclaimer": MEDICAL_DISCLAIMER,
        }), 400

    if model_pipeline is None:
        load_artifacts()
        if model_pipeline is None:
            return jsonify({
                "status": "error",
                "message": "ML model pipeline not loaded on server.",
                "disclaimer": MEDICAL_DISCLAIMER,
            }), 500

    df_single = pd.DataFrame([cleaned_data])
    prob = float(model_pipeline.predict_proba(df_single)[0][1])
    pred = int(model_pipeline.predict(df_single)[0])
    risk_tier = get_risk_tier(prob)
    top_factors = explain_prediction(cleaned_data, model_pipeline)

    return jsonify({
        "status": "success",
        "risk_probability": round(prob, 4),
        "risk_percentage": round(prob * 100, 1),
        "risk_level": risk_tier["level"],
        "risk_tier_description": risk_tier["description"],
        "prediction": pred,
        "prediction_label": "Presence of Heart Disease" if pred == 1 else "No Heart Disease Detected",
        "top_factors": top_factors,
        "patient_data": cleaned_data,
        "disclaimer": MEDICAL_DISCLAIMER,
    }), 200


@app.route("/health")
def health():
    """Liveness probe for cloud platform deployment."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "best_model": metrics_data.get("best_model", "Unknown"),
    }), 200


@app.errorhandler(404)
def not_found(e):
    return render_template("base.html", not_found=True, disclaimer=MEDICAL_DISCLAIMER), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
