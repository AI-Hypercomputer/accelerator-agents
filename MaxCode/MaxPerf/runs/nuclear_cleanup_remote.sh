#!/usr/bin/env bash
echo "=== SYSTEM ACCELERATOR LOCK PURGE ==="
# Kill all processes holding TPU device descriptors
sudo fuser -k /dev/accel* || true
# Stop ray clusters
~/vllm_env/bin/ray stop --force || true
# Kill legacy python processes
pkill -9 -f vllm || true
pkill -9 -f ray || true
pkill -9 -f python3 || true
echo "=== SYSTEM ACCELERATOR LOCK PURGE COMPLETED ==="
