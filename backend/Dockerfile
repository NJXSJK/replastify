# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# OS dependencies:
#   curl         — needed by HEALTHCHECK
#   libjpeg-dev  — Pillow JPEG decoding
#   zlib1g-dev   — Pillow PNG decoding
RUN apt-get update && apt-get install -y \
    curl \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-specific PyTorch wheels to keep Docker image size optimized (~800MB vs ~3.8GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# Health check — verifies the /health endpoint responds within 10s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# workers=1: model singleton lives in one process
# For production scale, use TorchServe or Triton instead
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
