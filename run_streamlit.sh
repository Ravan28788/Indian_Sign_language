#!/usr/bin/env bash
# Quick launcher for ISL Streamlit Application

cd "$(dirname "$0")"

# Configure cache directories within workspace
export MPLCONFIGDIR="$(pwd)/.cache/matplotlib"
export KERAS_HOME="$(pwd)/.cache/keras"
mkdir -p "$MPLCONFIGDIR" "$KERAS_HOME"

if [ -f ".venv/bin/streamlit" ]; then
    echo "Starting ISL Vision Pro in virtual environment..."
    ./.venv/bin/streamlit run streamlit_app.py
elif command -v streamlit &> /dev/null; then
    echo "Starting ISL Vision Pro using system Streamlit..."
    streamlit run streamlit_app.py
else
    echo "Streamlit is not yet installed. Please run:"
    echo "  pip install -r requirements.txt"
fi
