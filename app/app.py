import os
import numpy as np
import fastapi
import pydantic
import onnxruntime as ort
import traceback

# ----------------------------
# Initialize FastAPI
# ----------------------------
app = fastapi.FastAPI(title="Titanic Survival Predictor (ONNX)")

# ----------------------------
# Load ONNX model
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.onnx")

try:
    session = ort.InferenceSession(MODEL_PATH)
    print(f"ONNX model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load ONNX model at {MODEL_PATH}")
    raise e

# ----------------------------
# Define input schema
# ----------------------------
class InputData(pydantic.BaseModel):
    pclass: int
    age: float
    sibsp: int
    parch: int
    fare: float
    sex_male: int
    embarked_Q: int
    embarked_S: int

# ----------------------------
# Health check route
# ----------------------------
@app.get("/")
def root():
    return {"message": "Titanic ONNX model API is running!"}

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        # Arrange features in the correct order
        features = [
            data.pclass,
            data.age,
            data.sibsp,
            data.parch,
            data.fare,
            data.sex_male,
            data.embarked_Q,
            data.embarked_S,
        ]

        # Convert to numpy array with batch dimension
        input_array = np.array([features], dtype=np.float32)

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: input_array})

        # Return the probability of survival
        return {"prediction": float(outputs[0][0][0])}

    except Exception as e:
        print("Error during prediction:")
        print(traceback.format_exc())
        raise fastapi.HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )
