import sys
import os
import importlib

# ----------------------------
# Fix imports when running from test/ folder
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ----------------------------
# Import FastAPI app dynamically
# ----------------------------
app_module = importlib.import_module("app.app")
app = getattr(app_module, "app")

# ----------------------------
# FastAPI test imports
# ----------------------------
from fastapi.testclient import TestClient

client = TestClient(app)

# ----------------------------
# Test health endpoint
# ----------------------------
def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Titanic ONNX model API is running!"}

# ----------------------------
# Test valid prediction
# ----------------------------
def test_predict_valid():
    sample_input = {
        "pclass": 3,
        "age": 22.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "sex_male": 1,
        "embarked_Q": 0,
        "embarked_S": 1,
    }
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert isinstance(prediction, float)
    assert 0.0 <= prediction <= 1.0

# ----------------------------
# Test invalid input
# ----------------------------
def test_predict_invalid():
    # FastAPI raises 422 for invalid input
    response = client.post("/predict", json={"wrong": "data"})
    assert response.status_code == 422
