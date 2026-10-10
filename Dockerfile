# ── Multi-stage production build for ERC Analyzer ─────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

# Install system dependencies (curl, build tools, and libgomp1 OpenMP required by PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install CPU PyTorch followed by app requirements
COPY requirements.txt .
RUN pip install torch==2.5.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Pre-download and cache fast, lightweight cross-encoder reranker weights (~80MB)
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" || true

# Copy application source code and frontend
COPY . .

# Expose default port
EXPOSE 8000

# Start server using run.py which dynamically binds to Render's $PORT
CMD ["python", "run.py"]
