# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (build tools and libGL for OpenCV/PIL if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and application assets
COPY src/ src/
COPY scripts/ scripts/
COPY sample_images/ sample_images/
COPY checkpoints/ checkpoints/
COPY app.py .

# Expose Gradio default port
EXPOSE 7860

# Environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860"

# Launch NASA JPL Mission Control Telemetry Dashboard
CMD ["python", "app.py"]
