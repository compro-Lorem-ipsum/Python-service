# Base image menggunakan NVIDIA CUDA (GPU support)
# nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 include CUDA toolkit + cuDNN
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Non-interactive install
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Python runtime defaults
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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
RUN python -m pip install --upgrade pip setuptools wheel

# Install dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip uninstall -y opencv-python || true
# Copy seluruh project
COPY . .

# Expose port FastAPI
EXPOSE 8000

# Start FastAPI via uvicorn (GPU workloads usually prefer a single worker)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]