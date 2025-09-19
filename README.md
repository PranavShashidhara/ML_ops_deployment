# MLOPS Project

## Overview

This project demonstrates an end-to-end MLOps workflow using a machine learning model trained on the Titanic dataset. The focus is on production-ready deployment and reproducibility, showing how a model can move from development to production efficiently.

Key aspects of this project include:
- Model Development:
    - Training a machine learning model using PyTorch.
    - Exporting the model to ONNX for optimized inference.
- Containerization:
    - Packaging the application and model into a Docker container for consistent deployment across environments.
- Deployment & Serving:
    - Serving the model via FastAPI with ONNX Runtime for low-latency inference.
    - Exposing endpoints for predictions, ready for integration into other systems.
- MLOps Practices:
    - Designing the project for CI/CD pipelines to automate testing and deployment.
    - Organizing the repository for scalability and maintainability.
    - Using `.dockerignore` and proper folder structure to ensure clean, reproducible builds.
    - Adding automated tests for both the model and API endpoints.


File structure is as shown below: 
```bash 
    ML_ops_deployment/
        ├── app/
        │   ├── app.py                # FastAPI service
        │   └── titanic_model.onnx    # ONNX model
        ├── docker/
        │   ├── Dockerfile            # Dockerfile to run the model 
        │   └── requirements.txt      # lightweight libraries to just run the model 
        ├── requirements.txt          # Python dependencies
        ├── test/                     # Unit and API tests
        │   ├── test_model.py         # Model-level tests (single/batch predictions)
        │   └── test_app.py           # API endpoint tests (health, valid/invalid inputs)
        ├── .dockerignore             # files/folders to exclude from Docker
        ├── .gitignore                # To ignore unwanted files which should not be pushed to GitHub
        └── README.md                 # Project documentation
```

## Running the project
1. Install dependencies (optional for local testing): 
```bash 
> python -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt 
```
2. Run FastAPI locally: 
```bash 
> uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```
3. Access the API:

- Health check:
```bash
> curl http://localhost:8000/
# Output: {"message":"Titanic ONNX model API is running!"}
```
Predict survival probability:
```bash 
> curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"pclass":3,"age":22.0,"sibsp":1,"parch":0,"fare":7.25,"sex_male":1,"embarked_Q":0,"embarked_S":1}'
```
## Docker
Build and run Docker container: 
```bash 
# Build the Docker image (run only when code or dependencies change)
> docker build -t onnx-fastapi .

# Run the container (can be done multiple times)
> docker run -p 8000:8000 onnx-fastapi
```
> **Note:** Once the Docker image is built, you can reuse it to start the container multiple times without rebuilding.

The API will be available at http://localhost:8000/.

## Testing 
1. Run model-level tests and API endpoint tests using pytest: 
```bash 
> pytest -v test/ 
```
2. Expected outcomes: 
- Model tests: Check loading, single/batch predictions and output shapes. 
- API tests: 
    - Health endpiont (GET /)
    - Valid prediction (POST /predict with correct input)
    - Invalid prediction (POST /prediction with wrong input)
3. Where and when to run tests:
- Where: From the project root, where the test/ folder is located.
- When:
    - After modifying the model or retraining it.
    - After making changes to API endpoints or business logic.
    - Before committing code or triggering CI/CD pipelines.
4. Example commands for specific tests:
- Run model tests only:
```bash 
> pytest -v test/test_model.py
```
- Run API test only: 
```bash 
> pytest -v test/ 
```
> **Note:** When using 'pytest' with 'TestClient', the FastAPI app runs internally, so you don’t need to start 'uvicorn'. For manual testing with 'curl' or Postman, the app must be running via 'uvicorn'.
## Cheat Sheet (Common Commands)
| Action | Command | Location | When to Run |
| --- | --- | --- | --- |
| Create virtual environment | `python -m venv venv` | Project root | First time setting up the project |
| Activate virtual environment | `source venv/bin/activate` | Project root | Every time before running Python scripts locally |
| Install dependencies | `pip install -r requirements.txt` | Project root | After creating/activating venv, or when dependencies change |
| Run FastAPI locally | `uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload` | Project root | For local testing and development |
| Build Docker image | `docker build -t onnx-fastapi .` | Project root | Only when code or dependencies change |
| Run Docker container | `docker run -p 8000:8000 onnx-fastapi` | Project root | To test the API in Docker |
| Run model tests | `pytest -v test/test_model.py` | Project root | After any model-related changes |
| Run API tests | `pytest -v test/test_app.py` | Project root | After any API-related changes |
| Run all tests | `pytest -v test/` | Project root | To run all automated tests |
| Lint code with Black | `black . --exclude venv` | Project root | Before committing code or as part of CI/CD |