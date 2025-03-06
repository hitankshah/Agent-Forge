#!/bin/bash
# Simple launcher script for the platform

echo "Starting Agent Platform..."

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH!"
    echo "Please install Python 3.8 or later."
    exit 1
fi

# Check for arguments
UI_TYPE="integrated"
API_FLAG=""

if [ "$1" == "legacy" ]; then
    UI_TYPE="legacy"
fi

if [ "$2" == "api" ]; then
    API_FLAG="--api"
fi

echo "Launching with UI type: $UI_TYPE"

# Launch the platform
python3 unified_server.py --ui $UI_TYPE $API_FLAG

if [ $? -ne 0 ]; then
    echo "Failed to start platform!"
fi
