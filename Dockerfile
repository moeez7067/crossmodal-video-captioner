# Use Python 3.9 as base image
FROM python:3.9-slim

# Install system dependencies INCLUDING GIT
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/temp/uploads data/temp/frames data/temp/audio models

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"]