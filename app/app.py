import os
import numpy as np
import fastapi 
import pydantic
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort


# Initialize FastAPI
app = fastapi.FastAPI(title="ONNX Model API")

# Load ONNX model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.onnx")

session = ort.InferenceSession(MODEL_PATH)

# Define input schema
class InputData(pydantic.BaseModel):
    features: list[float]  # Example: [1.2, 3.4, 5.6]

@app.post("/predict")
def predict(data: InputData):
    # Convert input to numpy
    input_array = np.array(data.features, dtype=np.float32).reshape(1, -1)

    # Prepare ONNX input
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})

    # Assuming single output
    return {"prediction": outputs[0].tolist()}
