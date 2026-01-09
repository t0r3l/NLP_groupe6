# =============================================================================
# ULTRA-OPTIMIZED Dockerfile for RAG Historian
# Target: Minimal image size, CPU-only, fewest layers possible
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install all dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Single layer: Install build deps, uv, Python packages, cleanup - all in one
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --no-cache-dir uv && \
    rm -rf /var/lib/apt/lists/* /root/.cache

# Copy dependency files
COPY pyproject.toml Readme.md ./
COPY src_rag/ ./src_rag/

# Install Python dependencies with CPU-only PyTorch
# --no-cache: don't cache packages (smaller layer)
# --compile-bytecode: pre-compile for faster startup
# CPU index: avoids ~3GB of NVIDIA packages
RUN uv pip install --system --no-cache --compile-bytecode \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -e . && \
    # Remove unnecessary files from site-packages
    find /usr/local/lib/python3.11/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -name "*.pyo" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -name "*.so" -exec strip --strip-unneeded {} \; 2>/dev/null || true && \
    # Remove NVIDIA stubs if any leaked through
    rm -rf /usr/local/lib/python3.11/site-packages/nvidia* 2>/dev/null || true && \
    rm -rf /usr/local/lib/python3.11/site-packages/triton* 2>/dev/null || true

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# Labels for image metadata
LABEL maintainer="NLP Groupe 6" \
      version="1.0" \
      description="RAG Historian - Lightweight Streamlit App"

WORKDIR /app

# Single layer: Create user, install curl, set permissions
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

# Copy Python packages from builder (single layer)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (single layer with chown)
COPY --chown=appuser:appuser src_rag/ ./src_rag/
COPY --chown=appuser:appuser app.py config.yml ./
COPY --chown=appuser:appuser data/ ./data/

# Switch to non-root user
USER appuser

# Environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.fileWatcherType=none"]
