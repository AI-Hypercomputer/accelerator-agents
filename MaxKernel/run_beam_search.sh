#!/bin/bash
# Local wrapper to execute Beam Search orchestrator dry-runs

# Exit immediately if a command exits with a non-zero status
set -e

# Move to the MaxKernel project root directory
CDPATH= cd -- "$(dirname -- "$0")"

# Activate the python virtual environment from the autocomp clone
VENV_PATH="/usr/local/google/home/ligh/github/accelerator-agents/MaxKernel/.venv"
if [ -d "$VENV_PATH" ]; then
  echo "[Shell] Activating virtual environment: $VENV_PATH"
  source "$VENV_PATH/bin/activate"
else
  echo "[Error] Virtual environment not found at $VENV_PATH"
  exit 1
fi

# Set mock compiler environment variables
export MOCK_COMPILER=true
export GOOGLE_CLOUD_PROJECT="tpu-kernel-assist-sandbox"
export RAG_CORPUS="projects/tpu-kernel-assist-sandbox/locations/us-west1/ragCorpora/7991637538768945152"


echo "[Shell] Launching Beam Search local verification..."
python3 -u beam_search/run_beam_search.py "$@"

echo "[Shell] Beam Search verification finished."
