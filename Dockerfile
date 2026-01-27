FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (required for opencv, scipy, etc.)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports (optional: MLflow / API / Jupyter)
EXPOSE 5000 8080 8888

# Default command: run DVC pipeline
CMD ["dvc", "repro"]
