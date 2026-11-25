#!/usr/bin/env bash
# IndTranslate Startup Script
# Optimized for Linux (Arch / CachyOS) and all Unix environments

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "🇮🇳 Starting IndTranslate Full-Stack Platform"
echo "Breaking Language Barriers Across India"
echo "=========================================================="

# Environment optimizations (CachyOS / Arch / Linux performance)
export PYTHONPATH="$DIR"
export MPLCONFIGDIR="/tmp/matplotlib"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# Auto-link pip-installed CUDA / cuDNN libraries for RTX 4050 if available
NVIDIA_PIP_DIR="$DIR/.venv/lib/python3.11/site-packages/nvidia"
if [ -d "$NVIDIA_PIP_DIR" ]; then
    NVIDIA_LIBS="$(find "$NVIDIA_PIP_DIR" -maxdepth 2 -type d -name "lib" 2>/dev/null | paste -sd ":" -)"
    if [ -n "$NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
    fi
fi

# Ensure venv Python is present
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
    UVICORN_EXEC=".venv/bin/uvicorn"
else
    PYTHON_EXEC="python3"
    UVICORN_EXEC="uvicorn"
fi

# Check if model exists, if not, train it
if [ ! -f "backend/saved_models/isl_model.keras" ]; then
    echo "⚡ Initializing ISL Deep Learning model training..."
    $PYTHON_EXEC -m backend.app.services.isl_dataset_builder
fi

# Check if frontend is built
if [ ! -d "frontend/dist" ]; then
    echo "📦 Building React frontend..."
    npm run build --prefix frontend
fi

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "🚀 Launching FastAPI server on http://localhost:${PORT}"
echo "📖 Swagger API Docs available at http://localhost:${PORT}/docs"
echo "=========================================================="

exec $UVICORN_EXEC backend.app.main:app --host "$HOST" --port "$PORT" --reload
