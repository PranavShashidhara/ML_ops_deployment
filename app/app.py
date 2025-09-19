import os
import numpy as np
import fastapi
import pydantic
import onnxruntime as ort
import traceback

# Initialize FastAPI
app = fastapi.FastAPI(title="ONNX Model API")

# Load ONNX model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.onnx")

try:
    session = ort.InferenceSession(MODEL_PATH)
except Exception as e:
    print(f"Failed to load ONNX model at {MODEL_PATH}")
    raise e

# Define input schema
class InputData(pydantic.BaseModel):
    features: list[float]  # Example: [1.2, 3.4, 5.6]

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to numpy array and add batch dimension
        input_array = np.array(data.features, dtype=np.float32).reshape(1, -1)

        # Prepare ONNX input
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_array})

        # Return prediction
        return {"prediction": outputs[0].tolist()}

    except Exception as e:
        # Print full traceback to logs for debugging
        print("Error during prediction:")
        print(traceback.format_exc())

        # Return readable error message to client
        raise fastapi.HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
