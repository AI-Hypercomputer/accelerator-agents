---
title: "v039: TP=1 Vocab-Parallel CE - Batch Size Scaling (bs=32)"
date: "2026-06-04"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v039 - TP=1 Vocab-Parallel CE - Batch Size Scaling (bs=32)

## Hypothesis under test
- **Hypothesis**: In `v038` and `v037`, we found that algorithmic optimizations (RoPE fusion, Chunked XLA) crash the TPU due to Pallas VMEM limits or command queue stalls. In previous experiments (`v035`), we hit 34.6% MFU by simply increasing batch size to `bs=24` (seqlen 2048). To achieve the final >35% MFU goal, we can scale the local batch size even further to `bs=32` by reducing the sequence length to `L=1024` to avoid HBM OOMs. Higher arithmetic intensity from a larger batch dimension will maximize the matrix multiplication units and push MFU past 35%.
- **Mechanism**: Use the stable `v027` configuration (Splash + Dense XLA CE + no outer checkpoint) but set `--batch_size=32` and reduce `seqlen` in the fake data generator if necessary.
- **Predicted signal**: The model runs without OOMs, throughput scales, and MFU hits >35%.
- **Falsification criterion**: MFU plateaus or HBM OOMs occur even with reduced seqlen.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 32
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash Attention
  - Tokamax `implementation="xla"`
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Vocab-Parallel FSDP CE
  - Sequence Length: 1024

## Setup Steps
1. Fork codebase from `v026`/`v027`.
2. Edit `train.py` to change `seqlen` to 1024.
3. Launch via `gke-cluster-runner` with `--batch_size=32 --tp_parallelism=1 --use_splash=True`.

## Results
- **Status**: Failed (OOM)
- **Throughput**: N/A
- **Approx MFU**: N/A

## Verdict
Falsified. The workload crashed during XLA compilation with `RESOURCE_EXHAUSTED`. The model attempted to allocate 35.80 GB of HBM, exceeding the v6e chip's 31.25 GB capacity by 4.55 GB. Despite halving the sequence length to 1024, the memory footprint required for `bs=32` activations and gradients under pure `TP=1` FSDP scaling is strictly beyond hardware limits. Batch scaling has officially peaked at `bs=24`.
