FROM python:3.11-slim

WORKDIR /app

# System build dependencies for Prophet / scikit-learn / pystan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached separately from source)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source — the entire repo root is mounted as /app
# We copy the backend package under /app/backend so relative imports work
COPY . /app/backend

# Create data directory
RUN mkdir -p /app/backend/data

EXPOSE 8000

# Run as a package: uvicorn backend.main:app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
