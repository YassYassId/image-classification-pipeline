# Simple CPU-only Dockerfile for the CIFAR-10 DVC pipeline
FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps (git for DVC, build essentials for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list and install first (better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Ensure directories exist
RUN mkdir -p data/raw/cifar10 data/processed models metrics/plots

# Default command prints help; to run full pipeline: docker run --rm -v %cd%/data:/app/data image dvc repro
CMD ["bash", "-lc", "echo 'Container is ready. Mount your data/ and run: dvc repro'"]
