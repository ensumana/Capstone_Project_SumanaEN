# ============================================================
# Hospital Risk Intelligence API - Dockerfile
# ============================================================

FROM python:3.10-slim

LABEL maintainer="Hospital Analytics Team"
LABEL version="1.0.0"
LABEL description="Hospital Risk Intelligence API"

# ============================================================
# Working Directory
# ============================================================

WORKDIR /app

# ============================================================
# System Dependencies
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Copy requirements and install packages
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy project files
# ============================================================

COPY Phase5_APIIntegration ./Phase5_APIIntegration
COPY Utils ./Utils
COPY models ./models

# Optional if present
RUN mkdir -p /app/Outputs_Phase5


# ============================================================
# Create Logs Folder
# ============================================================

RUN mkdir -p /app/Phase5_APIIntegration/logs

# ============================================================
# Expose Port
# ============================================================

EXPOSE 8000

# ============================================================
# Health Check
# ============================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ============================================================
# Start API
# ============================================================

CMD ["uvicorn", "Phase5_APIIntegration.main:app", "--host", "0.0.0.0", "--port", "8000"]