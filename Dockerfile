# ── Multi-stage production build for ERC Analyzer ─────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

# Install system dependencies (curl for healthchecks, build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to avoid heavy CUDA packages (~2GB savings)
RUN pip install torch==2.5.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Copy dependencies list and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download and cache cross-encoder reranker weights during build
# This ensures container starts immediately without Hugging Face download delay in runtime
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')" || true

# Copy application source code and frontend
COPY . .

# Expose default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
