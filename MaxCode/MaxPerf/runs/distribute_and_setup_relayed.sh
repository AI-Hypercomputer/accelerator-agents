#!/usr/bin/env bash
# distribute_and_setup_relayed.sh — Setup & runner for maxperf-v6e-1.
set -ex

TPU_NAME="gvanica_google_com@maxperf-v6e-1"
ZONE="asia-northeast1-b"

PROJECT="tpu-prod-env-multipod"

echo "=== 1. Uploading files to worker ==="
gcloud alpha compute tpus tpu-vm scp \
    runs/patch_tpu_platform.py runs/patch_vllm_async_scheduling.py runs/setup_tpu_env_remote.sh runs/run_level0.sh runs/patch_vllm_weights.py runs/nuclear_cleanup_remote.sh \
    tools/run_numeric_equiv.py raw/numeric_ref/prompts.jsonl \
    "$TPU_NAME:~/" \
    --zone "$ZONE" \
    --project "$PROJECT" \
    --worker=0

echo "=== 2. Renaming, preparing scripts and cleaning up stale locks ==="
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone "$ZONE" \
    --project "$PROJECT" \
    --worker=0 \
    --command="mv ~/setup_tpu_env_remote.sh ~/setup_tpu_env.sh && chmod +x ~/setup_tpu_env.sh ~/patch_tpu_platform.py ~/patch_vllm_async_scheduling.py ~/run_level0.sh ~/patch_vllm_weights.py ~/nuclear_cleanup_remote.sh && ~/nuclear_cleanup_remote.sh" \
    -- -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /Users/gvanica/.ssh/google_compute_engine

echo "=== 3. Patching vLLM weight loaders ==="
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone "$ZONE" \
    --project "$PROJECT" \
    --worker=0 \
    --command="chmod +x ~/patch_vllm_weights.py && ~/patch_vllm_weights.py" \
    -- -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /Users/gvanica/.ssh/google_compute_engine

echo "=== 4. Orchestrating Ray head on worker ==="
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone "$ZONE" \
    --project "$PROJECT" \
    --worker=0 \
    --command="export PHASED_PROFILING_DIR=${PHASED_PROFILING_DIR:-''} && source ~/vllm_env/bin/activate && ray stop --force || true && ray start --head --port=6379 --disable-usage-stats" \
    -- -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /Users/gvanica/.ssh/google_compute_engine

sleep 5

echo "=== 5. Launching Level 0 Baseline Benchmark ==="
cmd=""
[ -n "${BLOCK_SIZE:-}" ] && cmd+="BLOCK_SIZE=$BLOCK_SIZE "
[ -n "${MAX_NUM_SEQS:-}" ] && cmd+="MAX_NUM_SEQS=$MAX_NUM_SEQS "
[ -n "${KV_CACHE_DTYPE:-}" ] && cmd+="KV_CACHE_DTYPE=$KV_CACHE_DTYPE "
[ -n "${MAX_MODEL_LEN:-}" ] && cmd+="MAX_MODEL_LEN=$MAX_MODEL_LEN "
[ -n "${TP_SIZE:-}" ] && cmd+="TP_SIZE=$TP_SIZE "
[ -n "${NUM_PROMPTS:-}" ] && cmd+="NUM_PROMPTS=$NUM_PROMPTS "
[ -n "${MAX_CONCURRENCY:-}" ] && cmd+="MAX_CONCURRENCY=$MAX_CONCURRENCY "
[ -n "${PHASED_PROFILING_DIR:-}" ] && cmd+="PHASED_PROFILING_DIR=$PHASED_PROFILING_DIR "
cmd+="~/run_level0.sh"

gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone "$ZONE" \
    --project "$PROJECT" \
    --worker=0 \
    --command="$cmd" \
    -- -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /Users/gvanica/.ssh/google_compute_engine

echo "=== Setup and Run initiated! ==="
