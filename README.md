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
    ├── .dockerignore             # files/folders to exclude from Docker
    ├── .gitignore                # To ignore unwanted files which should not be pushed to github
    └── README.md                 # project documentation
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
3. Build and run Docker container: 
```bash 
# Build the Docker image (run only when code or dependencies change)
> docker build -t onnx-fastapi .

# Run the container (can be done multiple times)
> docker run -p 8000:8000 onnx-fastapi
```
> **Note:** Once the Docker image is built, you can reuse it to start the container multiple times without rebuilding.

The API will be available at http://localhost:8000/.