# Dockerfile for Hugging Face Space (Docker type)

FROM python:3.11-slim

WORKDIR /app

# Install system deps (git, etc. optional but good to have)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first, install, then copy code (build cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the default Spaces port
EXPOSE 7860

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
