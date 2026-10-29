"""
HeartGuard Automated Test Suite
Covers web routes, form submissions, input validation, and REST API endpoints.
"""

import os
import json
import pytest
from app import app, MEDICAL_DISCLAIMER


@pytest.fixture
def client():
    """Create Flask test client configured for testing."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def healthy_patient():
    """Low-risk sample patient payload."""
    return {
        "age": 42,
        "sex": 0,
        "cp": 2,
        "trestbps": 118,
        "chol": 185,
        "fbs": 0,
        "restecg": 0,
        "thalach": 172,
        "exang": 0,
        "oldpeak": 0.2,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }


@pytest.fixture
def high_risk_patient():
    """High-risk sample patient payload."""
    return {
        "age": 67,
        "sex": 1,
        "cp": 4,
        "trestbps": 160,
        "chol": 286,
        "fbs": 0,
        "restecg": 2,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 2,
        "ca": 3,
        "thal": 7,
    }


def test_home_page(client):
    """Test that the landing page loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "HeartGuard" in html
    assert "Check My Risk" in html
    assert MEDICAL_DISCLAIMER in html


def test_about_page(client):
    """Test that the about page renders model benchmark metrics."""
    response = client.get("/about")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Benchmark Comparison" in html
    assert "UCI Cleveland Heart Disease" in html
    assert "Logistic Regression" in html
    assert "/api/predict" in html


def test_predict_form_get(client):
    """Test that the clinical form loads with all 13 inputs."""
    response = client.get("/predict")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'name="age"' in html
    assert 'name="trestbps"' in html
    assert 'name="chol"' in html
    assert 'name="thalach"' in html
    assert 'name="oldpeak"' in html
    assert 'name="cp"' in html
    assert 'name="ca"' in html
    assert 'name="thal"' in html
    assert "Evaluate Cardiovascular Risk" in html


def test_predict_form_post_success(client, healthy_patient):
    """Test submitting valid patient details via form."""
    response = client.post("/predict", data=healthy_patient)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Cardiovascular Risk Evaluation" in html
    assert "Predicted Risk" in html
    assert "Top Influencing Factors" in html
    assert MEDICAL_DISCLAIMER in html


def test_predict_form_post_high_risk(client, high_risk_patient):
    """Test submitting high-risk patient profile."""
    response = client.post("/predict", data=high_risk_patient)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "High Risk" in html
    assert "Elevated Cardiovascular Risk Detected" in html


def test_predict_form_validation_failure(client):
    """Test form submission with missing required parameters."""
    incomplete_data = {"age": 50, "sex": 1}
    response = client.post("/predict", data=incomplete_data)
    assert response.status_code == 400
    html = response.get_data(as_text=True)
    assert "Please correct the following input issues" in html
    assert "Missing field" in html


def test_predict_form_out_of_bounds(client, healthy_patient):
    """Test validation rejection of non-physiological values."""
    invalid_patient = healthy_patient.copy()
    invalid_patient["trestbps"] = 999  # BP beyond 240 mm Hg
    response = client.post("/predict", data=invalid_patient)
    assert response.status_code == 400
    html = response.get_data(as_text=True)
    assert "outside valid physiological range" in html


def test_api_predict_success(client, healthy_patient):
    """Test JSON REST API with healthy patient."""
    response = client.post(
        "/api/predict",
        data=json.dumps(healthy_patient),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert "risk_level" in data
    assert data["risk_level"] in ["Low", "Moderate", "High"]
    assert "top_factors" in data
    assert data["disclaimer"] == MEDICAL_DISCLAIMER


def test_api_predict_high_risk(client, high_risk_patient):
    """Test JSON REST API with high-risk patient."""
    response = client.post(
        "/api/predict",
        data=json.dumps(high_risk_patient),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["risk_level"] == "High"
    assert data["prediction"] == 1
    assert data["risk_probability"] > 0.70


def test_api_predict_invalid_json(client):
    """Test API error handling when non-JSON payload is sent."""
    response = client.post("/api/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"


def test_api_predict_missing_fields(client):
    """Test API input validation with missing fields."""
    response = client.post(
        "/api/predict",
        data=json.dumps({"age": 45}),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "errors" in data
    assert len(data["errors"]) > 0


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
