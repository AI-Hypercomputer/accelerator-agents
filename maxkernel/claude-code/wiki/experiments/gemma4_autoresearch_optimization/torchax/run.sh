#!/usr/bin/env bash
# Baseline run for Gemma 4 E4B on TPU v6e-8, via torchax.
#
# Usage:
#   bash run.sh                              # default 20 steps, profile steps 5..7
#   bash run.sh --steps 50 --seq_len 4096    # override any train.py flag
#
# Every extra flag is forwarded verbatim to `train.py`. Do not edit this
# script to change run parameters — edit the CLI.
#
# UNTESTED. See the header of train.py.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# wiki repo root = ../../../.. from this folder.
WIKI_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

TS="$(date +%Y-%m-%d-%H%M%S)"
PROFILE_DIR="${WIKI_ROOT}/raw/profiles/${TS}-gemma4-baseline"
HLO_DIR="${PROFILE_DIR}/hlo"
mkdir -p "$PROFILE_DIR" "$HLO_DIR"

# TPU / XLA flags. Source: wiki/sources/2026-xprof-mcp-tpu-optimization.md.
# Latency-hiding scheduler is the one flag that's universally safe to
# enable; the async-collective + megacore-fusion flags have workload-
# dependent effects and are left to individual hypotheses to flip on.
export XLA_FLAGS="${XLA_FLAGS:-} \
  --xla_dump_to=${HLO_DIR} \
  --xla_dump_hlo_as_text \
  --xla_tpu_enable_latency_hiding_scheduler=true"

# VMEM bump (TPU v6e). Safe default per the xprof-mcp TPU optimization
# guide. Bump higher if you hit VMEM spill in the op-stats tab.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} \
  --xla_tpu_scoped_vmem_limit_kib=131072"

# Shush some HF warnings.
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[run.sh] TS=${TS}"
echo "[run.sh] PROFILE_DIR=${PROFILE_DIR}"
echo "[run.sh] HLO_DIR=${HLO_DIR}"
echo "[run.sh] XLA_FLAGS=${XLA_FLAGS}"
echo "[run.sh] LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS}"

python "${SCRIPT_DIR}/train.py" \
  --profile_dir "${PROFILE_DIR}" \
  --profile_steps 5 6 7 \
  "$@"
