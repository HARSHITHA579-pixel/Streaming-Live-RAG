# Dockerfile for Streaming RAG Service
# Python 3.11 slim image for a lightweight and clean container footprint

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system-level dependencies for building native packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, corpus, and evaluation harnesses
COPY app/ /app/app/
COPY corpus/ /app/corpus/
COPY data/ /app/data/
COPY dashboard/ /app/dashboard/
COPY eval/ /app/eval/
COPY docs/ /app/docs/

# Create logs directory for telemetry output
RUN mkdir -p /app/logs

# Expose FastAPI / WebSocket port
EXPOSE 8000

# Default entry point (can be overridden for corpus indexing or evaluation)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
