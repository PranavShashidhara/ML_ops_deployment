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
    ├── requirements.txt          # Python dependencies
    ├── Dockerfile                # Docker configuration
    ├── .dockerignore             # files/folders to exclude from Docker
    └── README.md                 # project documentation
```

Run the project 
1. Install dependencies 
