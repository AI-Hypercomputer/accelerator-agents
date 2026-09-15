#!/usr/bin/env bash
# runs/phase0-best.sh — Best Phase 0 (Level 0) tuned configuration for Qwen2.5-Coder-3B-Instruct.
# Expose the tuned environment variables and server parameters.

export BLOCK_SIZE=256
export KV_CACHE_DTYPE="fp8"
export TENSOR_PARALLEL_SIZE=8
export MAX_NUM_SEQS=128
export MAX_MODEL_LEN=8192

echo "Best Phase 0 Config:"
echo "  BLOCK_SIZE: $BLOCK_SIZE"
echo "  KV_CACHE_DTYPE: $KV_CACHE_DTYPE"
echo "  TENSOR_PARALLEL_SIZE: $TENSOR_PARALLEL_SIZE"
echo "  MAX_NUM_SEQS: $MAX_NUM_SEQS"
echo "  MAX_MODEL_LEN: $MAX_MODEL_LEN"
