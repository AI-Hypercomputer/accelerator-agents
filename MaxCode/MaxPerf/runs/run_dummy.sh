#!/bin/bash
# run_dummy.sh — Serving Qwen3-Coder-480B-A35B-Instruct-FP8 with dummy weights for MaxPerf TPU profiling.
# This script runs without loading a real checkpoint, enabling fast compilation and profiling checks.

set -e

# Hardware configuration
TPU_TP_SIZE=${TPU_TP_SIZE:-8} # Shard across chips of TPU VM
PORT=8000

# Performance profiling configurations
export XLA_FLAGS="--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
                  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
                  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"

echo "====================================================================="
echo "Serving Qwen3-Coder-480B-A35B-Instruct-FP8 in DUMMY mode on TPU..."
echo "This executes performance optimization and profiling without checkpoints."
echo "====================================================================="

python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
    --load-format dummy \
    --tensor-parallel-size ${TPU_TP_SIZE} \
    --port ${PORT} \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --worker-use-ray \
    --distributed-executor-backend ray
