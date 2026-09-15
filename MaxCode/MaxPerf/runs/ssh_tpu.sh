#!/usr/bin/env bash
# ssh_tpu.sh — SSH helper to connect to the maxperf-v6e-1 TPU VM.
# Usage:
#   ./runs/ssh_tpu.sh             # Interactive shell
#   ./runs/ssh_tpu.sh "command"   # Run command non-interactively

set -e

TPU_NAME="gvanica_google_com@maxperf-v6e-1"
ZONE="asia-northeast1-b"

PROJECT="tpu-prod-env-multipod"

if [ $# -eq 0 ]; then
  echo "Connecting to TPU VM ($TPU_NAME) in interactive session..."
  exec gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone "$ZONE" \
    --project "$PROJECT" \
    --worker="0" \
    -- -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /Users/gvanica/.ssh/google_compute_engine
else
  CMD="$1"
  echo "Executing command on TPU VM ($TPU_NAME): $CMD"
  exec gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone "$ZONE" \
    --project "$PROJECT" \
    --worker="0" \
    --command "$CMD" \
    -- -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /Users/gvanica/.ssh/google_compute_engine
fi
