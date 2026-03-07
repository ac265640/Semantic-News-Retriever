FROM python:3.11-slim

LABEL maintainer="Trademarkia AI Assignment"
LABEL description="Semantic Search API — 20 Newsgroups with fuzzy clustering and semantic cache"

# ── System dependencies ────────────────────────────────────────────────────────
# build-essential is required for compiling some numpy/scipy dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching.
# If requirements.txt hasn't changed, pip install is not re-run on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ─────────────────────────────────────────────────────────
COPY src/ ./src/
COPY scripts/ ./scripts/

# Pre-trained models and corpus data are expected to be mounted from host
# (./data → /app/data) via docker-compose volume. This avoids re-running
# setup_corpus.py (which downloads and embeds 18k documents) on every build.
# If running standalone: run setup_corpus.py inside the container first.

# ── Expose the API port ────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Single uvicorn startup command (as specified in assignment) ────────────────
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
