#!/bin/bash
set -e

# Check for models directory and initialize
if [ "$DOWNLOAD_MODELS" = "true" ]; then
  echo "Checking for local models..."
  
  # Run the model downloader
  if [ -f "/app/download_models.py" ]; then
    echo "Running model downloader..."
    python /app/download_models.py --from-config
  else
    echo "Model downloader script not found!"
  fi
fi

# Download smaller model by default for faster startup
if [ ! -d "/app/models/phi-2" ]; then
  echo "Downloading minimal phi-2 model for fast startup..."
  python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/phi-2', local_dir='/app/models/phi-2', ignore_patterns=['*.pt', 'adapter_*.safetensors'])"
fi

# Check for Llama-2-7b model and download if necessary
if [ ! -d "/app/models/llama-2-7b" ]; then
  echo "Downloading Llama-2-7b model..."
  python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-2-7b-chat-hf', local_dir='/app/models/llama-2-7b', ignore_patterns=['*.pt', 'adapter_*.safetensors'])"
fi

# Execute the main command
echo "Starting Agent Platform..."
exec python unified_server.py --api "$@"
