---
title: "v022: TP=1 Vocab-Parallel CE using chunked_xla"
date: "2026-06-03"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v022 - TP=1 Vocab-Parallel CE using chunked_xla

## Objective
The objective of this experiment is to bypass the Pallas kernel fragmentation issue (which was causing massive memory overhead like 16.87 GB fragmentation) when using Vocab-Parallel Cross Entropy. By explicitly setting the implementation to `chunked_xla`, we force GSPMD to use standard JAX operations which natively support the `("fsdp", "tp")` vocab sharding configured in `v021`, thus avoiding full materialization without invoking the Pallas fallback.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 4
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash
  - Tokamax chunked_xla (`implementation="chunked_xla"`)
  - nnx.remat
  - nnx.scan
  - Fused RoPE
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Forked the `v021-tp1-vocab-parallel-bs4` setup to `wiki/experiments/qwen3_ag_autoresearch_optimization/jax/.repo/2026-06-03-v022-tp1-chunked-xla-bs4/`.
2. Extracted `train.py` from the previous Docker image (`<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260603-v021-tp1-vocab-parallel-bs4`).
3. In `train.py`, updated the `tokamax.linear_softmax_cross_entropy_loss` call to include `implementation="chunked_xla"`.
4. Updated `Dockerfile` to use the `v021` image as a base and properly copy the modified `train.py`.
5. Built and pushed the Docker image as `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260603-v022-tp1-chunked-xla-bs4`.
