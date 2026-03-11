#!/bin/bash

if ! command -v python3.11 &> /dev/null
then
    echo "python3.11 could not be found, please install it."
    exit
fi

VENV_DIR="$HOME/maxcode_venv"

# Create virtual environment if it doesn't exist or is incomplete
RECREATE_VENV=0
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  RECREATE_VENV=1
else
  VENV_PYTHON_VERSION=$("$VENV_DIR"/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [ "$VENV_PYTHON_VERSION" != "3.11" ]; then
    echo "Detected Python $VENV_PYTHON_VERSION in virtual environment, expected 3.11. Recreating..."
    rm -rf "$VENV_DIR"
    RECREATE_VENV=1
  else
    echo "Virtual environment '$VENV_DIR' with Python 3.11 already exists."
  fi
fi

if [ "$RECREATE_VENV" -eq 1 ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  if ! python3.11 -m venv "$VENV_DIR"; then
    echo "Failed to create virtual environment with ensurepip. This may be due to missing python3.11-venv package."
    echo "Attempting to create virtual environment with --without-pip..."
    if ! python3.11 -m venv --without-pip "$VENV_DIR"; then
      echo "Failed to create virtual environment without pip."
      exit 1
    else
      source "$VENV_DIR"/bin/activate
      echo "Installing pip..."
      curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
      python get-pip.py
      rm get-pip.py
      echo "Virtual environment '$VENV_DIR' created."
    fi
  else
    echo "Virtual environment '$VENV_DIR' created."
  fi
fi

# Activate virtual environment
source "$VENV_DIR"/bin/activate

# Install dependencies
pip install --upgrade pip --index-url https://pypi.org/simple
pip install --upgrade google-genai numpy google-adk absl-py faiss-cpu torch flax jax[cpu] --index-url https://pypi.org/simple

# Check for GOOGLE_API_KEY
if [ -z "$GOOGLE_API_KEY" ]; then
  echo "Warning: GOOGLE_API_KEY environment variable not set. Please set it to use Gemini models."
fi

echo "Dependencies installed."
echo "To activate: source $VENV_DIR/bin/activate"
