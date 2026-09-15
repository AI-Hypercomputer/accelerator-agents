#!/usr/bin/env bash
# setup_tpu_env_remote.sh — Script to install vLLM and tpu-inference on Cloud TPU VM.
set -ex

VENV_DIR="$HOME/vllm_env"

echo "=== Creating Virtual Environment with Python 3.11 ==="
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== Configuring Pip to use CPU index globally ==="
pip config set global.extra-index-url https://download.pytorch.org/whl/cpu

echo "=== Configuring Version Overrides ==="
export VLLM_VERSION_OVERRIDE=0.12.0

echo "=== Upgrading Pip and Installing Prerequisite Packages ==="
pip install --upgrade pip setuptools wheel

echo "=== Pre-installing PyTorch CPU-only ==="
pip install torch torchvision

echo "=== Installing CMake, Ninja & setuptools-scm ==="
pip install cmake ninja setuptools-scm

echo "=== Cloning Repositories ==="
cd "$HOME"
if [ ! -d "tpu-inference" ]; then
  git clone https://github.com/vllm-project/tpu-inference.git "tpu-inference"
fi
cd "$HOME/tpu-inference"
git checkout baa83f8ab58ce00c7ca4e7cb85d12b0d21ace0e8
git clean -fdx

cd "$HOME"
if [ ! -d "vllm" ]; then
  git clone https://github.com/vllm-project/vllm.git "vllm"
fi
cd "$HOME/vllm"
git checkout 963dc0b865a3b6011fde7e0d938f86245dccbfac
git clean -fdx

echo "=== Installing tpu-inference ==="
cd "$HOME/tpu-inference"
pip install -e .

echo "=== Installing vllm ==="
cd "$HOME/vllm"
# Patch license metadata to be PEP 621 compliant under newer setuptools
sed -i 's/license = "Apache-2.0"/license = {text = "Apache-2.0"}/g' pyproject.toml
sed -i '/license-files = /d' pyproject.toml
export VLLM_TARGET_DEVICE=tpu
pip install -e . --no-build-isolation

echo "=== Patching Circular Imports & Import Paths in site-packages ==="
chmod +x "$HOME/patch_tpu_platform.py" "$HOME/patch_vllm_async_scheduling.py"
python3 "$HOME/patch_tpu_platform.py"
python3 "$HOME/patch_vllm_async_scheduling.py"

echo "=== Verification ==="
python3 -c "import vllm; print('vLLM version:', vllm.__version__)"
python3 -c "import tpu_inference; print('tpu-inference imported successfully')"

echo "=== TPU environment setup complete! ==="
