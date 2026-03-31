FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY data/download_dataset.py ./data/download_dataset.py

# Create data and output directories
RUN mkdir -p data/raw data/processed outputs/models outputs/logs outputs/figures

# Default command: show help
CMD ["python", "src/train.py", "--help"]
