# === Stage 1: Builder ===
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ONLY_BINARY=:all: \
    PYTHONOPTIMIZE=2

WORKDIR /app

# --- System deps needed for building wheels ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean && apt-get autoclean && apt-get autoremove -y

# --- Install uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# --- Copy dependency metadata ---
COPY pyproject.toml uv.lock* ./

# --- Export dependencies (excluding dev/docs/notebooks groups) ---
RUN /root/.local/bin/uv export \
    --format requirements-txt \
    # --no-hashes \
    --no-group dev \
    --no-group docs \
    --no-group notebooks \
    # --no-group torch-cpu \
    > requirements.txt

# --- Remove problematic packages that are not needed for CPU-only execution ---
# nvidia-* packages: GPU CUDA packages (~2-3GB) not needed for CPU-only environments
# RUN grep -iv '^nvidia-' requirements.txt | grep -v '^--hash' > requirements-clean.txt && \
#     mv requirements-clean.txt requirements.txt
# RUN grep -ivE '^torch' requirements.txt > requirements-clean.txt && mv requirements-clean.txt requirements.txt
# RUN grep -ivE '^(torch|torchvision|torchaudio|triton|nvidia-)' requirements.txt > requirements-clean.txt && mv requirements-clean.txt requirements.txt

# Create a constraints file to avoid installing nvidia packages
# RUN printf "nvidia-cublas-cu12==0\nnvidia-cuda-*==0\nnvidia-*==0\n" > constraints.txt
# RUN printf "nvidia-cublas-cu12==0\nnvidia-cuda-cupti-cu12==0\nnvidia-cuda-nvrtc-cu12==0\nnvidia-cuda-runtime-cu12==0\nnvidia-cudnn-cu12==0\nnvidia-cufft-cu12==0\nnvidia-curand-cu12==0\nnvidia-cusolver-cu12==0\nnvidia-cusparse-cu12==0\nnvidia-nvjitlink-cu12==0\nnvidia-nvtx-cu12==0\n" > constraints.txt

# Install CPU-only torch first to avoid large CUDA packages being pulled in
# RUN /root/.local/bin/uv pip install --system --no-cache-dir \
#     --index-url https://download.pytorch.org/whl/cpu \
#     torch torchvision torchaudio

# --- Install all dependencies with CPU-only PyTorch index ---
# Using --index-url ensures torch installs from CPU-only wheels (170MB vs 2.5GB+ with CUDA)
RUN /root/.local/bin/uv pip install --system --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

# --- Cleanup build artifacts (as root before switching user) ---
RUN find /usr/local/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    # find /usr/local/lib/python3.12/site-packages -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type f -name "*.py[co]" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name "*.so" -exec strip {} + 2>/dev/null || true && \
    rm -rf /root/.cache /tmp/*

# === Stage 2: Runtime ===
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# --- Minimal runtime deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean && apt-get autoclean && apt-get autoremove -y

# --- Copy dependencies from builder (includes CPU-only torch) ---
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# --- Copy app code ---
COPY src/ ./src/
COPY data/ ./data/

# --- Non-root user ---
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# --- Minimal NLTK data ---
RUN python -m nltk.downloader punkt stopwords punkt_tab

EXPOSE ${PORT}

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
