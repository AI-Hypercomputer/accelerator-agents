#!/usr/bin/env bash
# distribute_and_setup.sh — Coordinates multi-host setup & execution using sequential connection spacing.
set -ex

TPU_NAME="maxperf-v6e-1"
ZONE="asia-northeast1-b"
PROJECT="tpu-prod-env-multipod"


echo "=== 1. Copying files to all workers sequentially ==="
for i in 0; do
  echo "Copying files to worker $i..."
  gcloud alpha compute tpus tpu-vm scp runs/patch_tpu_platform.py runs/patch_vllm_async_scheduling.py runs/setup_tpu_env_remote.sh runs/run_level0.sh \
      ${TPU_NAME}:~/ \
      --zone "$ZONE" \
      --project "$PROJECT" \
      --worker=$i
  sleep 2
done

echo "=== 2. Renaming and configuring files on all workers ==="
for i in 0; do
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
      --zone "$ZONE" \
      --project "$PROJECT" \
      --worker=$i \
      --command="mv ~/setup_tpu_env_remote.sh ~/setup_tpu_env.sh && chmod +x ~/setup_tpu_env.sh ~/patch_tpu_platform.py ~/patch_vllm_async_scheduling.py ~/run_level0.sh" &
  sleep 2
done
wait

echo "=== 3. Running CLEAN compilation and setup on all workers ==="
for i in 0; do
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
      --zone "$ZONE" \
      --project "$PROJECT" \
      --worker=$i \
      --command="rm -rf ~/vllm_env ~/tpu-inference ~/vllm && ~/setup_tpu_env.sh" &
  sleep 3
done
wait

echo "=== 4. Launching Level 0 Baseline Run on all workers ==="
for i in 0; do
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
      --zone "$ZONE" \
      --project "$PROJECT" \
      --worker=$i \
      --command="~/run_level0.sh" &
  sleep 3
done
wait
