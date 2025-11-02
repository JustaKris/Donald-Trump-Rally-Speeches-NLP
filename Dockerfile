# === Stage 1: Builder ===
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- System deps needed for building wheels ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv (fast Python package installer) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# --- Ensure uv is in PATH ---
# ENV PATH="/root/.local/bin:$PATH"

# --- Copy dependency metadata ---
COPY pyproject.toml uv.lock* ./

# --- Export dependencies (excluding dev/docs/notebooks groups) ---
RUN /root/.local/bin/uv export \
    --format requirements-txt \
    --no-group dev \
    --no-group docs \
    --no-group notebooks \
    > requirements.txt

# --- Install dependencies from requirements.txt ---
RUN /root/.local/bin/uv pip install --system -r requirements.txt

# --- Cleanup build artifacts (as root before switching user) ---
RUN find /usr/local/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type f -name "*.py[co]" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
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
    && rm -rf /var/lib/apt/lists/*

# --- Copy dependencies from builder (includes CPU-only torch) ---
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# --- Copy app code ---
COPY src/ ./src/
COPY static/ ./static/
COPY data/ ./data/

# --- Non-root user ---
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# --- Minimal NLTK data ---
RUN python -m nltk.downloader punkt stopwords punkt_tab

EXPOSE ${PORT}

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
