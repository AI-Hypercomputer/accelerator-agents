#!/usr/bin/env bash
# Native-JAX Gemma 4 E4B trainer wrapper. Mirrors torchax/run.sh.
#
# Usage: from this folder,
#   ./run.sh --batch_size 1 --seq_len 1024 --steps 20
# Extra args are forwarded to train.py.
set -euo pipefail

WIKI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.."/.. && pwd)"
cd "$(dirname "${BASH_SOURCE[0]}")"

export XLA_FLAGS="${XLA_FLAGS:-}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-}"

PY="${PY:-/home/alekseyv_google_com/miniconda3/envs/gemma4_py313/bin/python}"
exec "$PY" -m train "$@"
