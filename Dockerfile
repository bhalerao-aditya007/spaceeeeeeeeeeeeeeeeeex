# ── Build stage: install dependencies ────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps needed for scipy/numpy compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller image)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install other deps (skip torch/torchvision since already installed above)
COPY requirements_web.txt .
RUN pip install --no-cache-dir matplotlib>=3.7.0 && \
    pip install --no-cache-dir -r requirements_web.txt || true

# ── Runtime stage ────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install redis-server
RUN apt-get update && apt-get install -y --no-install-recommends redis-server && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Cloud Run sets PORT env var
ENV PORT=8080
EXPOSE 8080

# Start the server and redis
CMD ["bash", "-c", "redis-server --daemonize yes && python -u interface/app.py"]
