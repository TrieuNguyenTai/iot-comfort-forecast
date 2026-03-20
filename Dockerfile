# Dockerfile - IoT Comfort Index Forecasting System
FROM python:3.10-slim

# Install system dependencies for Tkinter + Matplotlib GUI
RUN apt-get update && apt-get install -y \
    python3-tk \
    tk-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Default command: run data collection pipeline
CMD ["python", "src/1_thu_thap_du_lieu.py"]