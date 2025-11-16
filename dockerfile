# Base image menggunakan Ubuntu untuk compatibility penuh
FROM ubuntu:22.04

# Non-interactive install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python dan system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libopenblas-dev \
    liblapack-dev \
    ffmpeg \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link untuk python3
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install numpy first
RUN pip3 install --no-cache-dir "numpy==1.24.3" "opencv-python-headless==4.8.1.78"

# Install ONNX Runtime dengan version yang stabil
RUN pip3 install --no-cache-dir "onnxruntime==1.14.1"

# Install remaining requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy seluruh project
COPY . .

# Expose port FastAPI
EXPOSE 8000

# Start FastAPI via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]