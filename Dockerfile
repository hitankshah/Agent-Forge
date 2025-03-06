FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file separately to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Create necessary directories
RUN mkdir -p data/agents data/deployments data/logs config models

# Download free models during image build
ENV HF_HOME=/app/models/huggingface
ENV TRANSFORMERS_CACHE=/app/models/transformers

# Create model download script
RUN echo '#!/bin/bash \n\
python -c "from huggingface_hub import snapshot_download; \
snapshot_download(\"google/gemma-2b\", local_dir=\"/app/models/gemma-2b\"); \
snapshot_download(\"microsoft/phi-2\", local_dir=\"/app/models/phi-2\"); \
snapshot_download(\"mistralai/Mistral-7B-v0.1\", local_dir=\"/app/models/mistral-7b\")"' > /app/download_models.sh && \
chmod +x /app/download_models.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    USE_LOCAL_MODELS=true \
    LOCAL_MODELS_DIR=/app/models

# Expose ports for Streamlit and API
EXPOSE 8501 8000

# Entrypoint script
RUN echo '#!/bin/bash \n\
# Check if models exist, download if specified \n\
if [ "$DOWNLOAD_MODELS" = "true" ]; then \n\
  echo "Downloading models..." \n\
  /app/download_models.sh \n\
fi \n\
# Launch the application \n\
exec python unified_server.py --api "$@"' > /app/entrypoint.sh && \
chmod +x /app/entrypoint.sh

# Entry point - uses unified_server.py to start both UI and API
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - can be overridden
CMD ["--ui", "integrated", "--port", "8501", "--api-port", "8000"]
