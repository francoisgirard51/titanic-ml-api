# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and training data
COPY src/ ./src/
COPY data/ ./data/

# Train the model inside the image
RUN python src/train.py

# Expose the FastAPI port
EXPOSE 8000

# Launch the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
