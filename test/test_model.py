import os
import numpy as np
import onnxruntime as ort
import argparse


# ----------------------------
# Helper to get ONNX session
# ----------------------------
def get_model_session() -> ort.InferenceSession:
    """
    Loads and returns the ONNX model session.
    """
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "app", "titanic_model.onnx")
    return ort.InferenceSession(MODEL_PATH)


# ----------------------------
# Single prediction function
# ----------------------------
def predict_titanic_survival(features: list[float]) -> float:
    """
    Predicts Titanic survival probability for a single input.
    """
    session = get_model_session()
    input_array = np.array([features], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_array})
    return float(outputs[0][0][0])


# ----------------------------
# Batch prediction function
# ----------------------------
def predict_batch(batch_features: list[list[float]]) -> np.ndarray:
    """
    Predicts Titanic survival probabilities for a batch of inputs.
    """
    session = get_model_session()
    input_array = np.array(batch_features, dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_array})
    return outputs[0]  # numpy array of shape (batch_size, 1)


# ----------------------------
# ===========================
# Model-level tests using both functions
# ===========================
def test_model_loads():
    session = get_model_session()
    assert session is not None
    assert len(session.get_inputs()) > 0
    assert len(session.get_outputs()) > 0


def test_inference_single():
    prediction = predict_titanic_survival([3, 22.0, 1, 0, 7.25, 1, 0, 1])
    assert 0.0 <= prediction <= 1.0


def test_batch_inference():
    batch_input = [
        [3, 22.0, 1, 0, 7.25, 1, 0, 1],
        [1, 38.0, 1, 0, 71.2833, 0, 0, 1],
        [3, 26.0, 0, 0, 7.925, 1, 1, 0],
    ]
    output = predict_batch(batch_input)
    assert output.shape == (3, 1)
    for prob in output:
        assert 0.0 <= prob[0] <= 1.0


def test_shapes_match_session():
    session = get_model_session()
    single_output = predict_batch([[3, 22.0, 1, 0, 7.25, 1, 0, 1]])
    assert single_output.shape == (1, 1)
    batch_output = predict_batch(
        [[3, 22.0, 1, 0, 7.25, 1, 0, 1], [1, 38.0, 1, 0, 71.2833, 0, 0, 1]]
    )
    assert batch_output.shape == (2, 1)


# ----------------------------
# Direct callable test
# ----------------------------
if __name__ == "__main__":
    print(
        "Single prediction:", predict_titanic_survival([3, 22.0, 1, 0, 7.25, 1, 0, 1])
    )
    batch_input = [[3, 22.0, 1, 0, 7.25, 1, 0, 1], [1, 38.0, 1, 0, 71.2833, 0, 0, 1]]
    print("Batch prediction:\n", predict_batch(batch_input))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX Titanic model")
    parser.add_argument(
        "--model",
        type=str,
        default="titanic_model.onnx",
        help="Name of the ONNX model file in app/ directory",
    )
    args = parser.parse_args()

    # Sample features
    sample_features = [3, 22.0, 1, 0, 7.25, 1, 0, 1]

    # Run prediction
    prediction = predict_titanic_survival(sample_features, model_name=args.model)
    print(f"Using model: {args.model}")
    print("Prediction raw output:", prediction)
