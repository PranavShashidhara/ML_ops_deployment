# Use a lightweight Python image 
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements first for caching
COPY docker/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

COPY app ./app

# Command to run the FASTAPI application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]