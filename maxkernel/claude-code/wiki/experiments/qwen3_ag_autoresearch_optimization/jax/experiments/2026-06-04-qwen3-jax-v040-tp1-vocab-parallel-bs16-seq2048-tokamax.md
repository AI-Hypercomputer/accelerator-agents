---
title: "v040: TP=1 Vocab-Parallel Tokamax Pallas CE (bs=16)"
date: "2026-06-04"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v040 - TP=1 Vocab-Parallel Tokamax Pallas CE (bs=16)

## Hypothesis under test
- **Hypothesis**: In `v039`, scaling batch size to 32 caused an HBM OOM during compilation because the dense XLA cross-entropy loss materializes the enormous `[B*L, V]` intermediate tensor (`[32768, 151936]`) on the FSDP mesh. Earlier, in `v037`, we tried `chunked_xla` to avoid this, but it stalled the command queue. The final path to >35% MFU is to use Tokamax's `pallas` implementation for the CE loss. The Pallas CE kernel computes the loss in a single fused block without materializing the full logits tensor, drastically reducing HBM requirements and eliminating the `data formatting` copy overhead.
- **Mechanism**: Set Tokamax implementation to `"pallas"`. Use `bs=16` and `L=2048`.
- **Predicted signal**: The model compiles without OOM, runs efficiently, and MFU hits >35% due to reduced memory overhead and increased algorithmic efficiency.
- **Falsification criterion**: Pallas CE crashes or is slower than dense XLA CE.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 16
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash Attention
  - Tokamax `implementation="pallas"`
  - Inner `nnx.remat`
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Fork codebase from `v026`.
2. Edit `train.py` to set `bs=16` and `L=2048`.
3. Edit `model/modeling_qwen3.py` to change `tokamax` implementation from `"xla"` to `"pallas"`.
4. Launch via `gke-cluster-runner`.

## Results
- **Status**: Crashed
- **Throughput**: N/A
- **Approx MFU**: N/A

## Verdict
**Invalid / Falsified**. The `v040` (and `v3`/`v4` iterations) crashed during `jax.jit` compilation with `NotImplementedError: Mosaic kernels cannot be automatically partitioned. Please wrap the call in a shard_map.` Pallas kernels are opaque custom calls and cannot be auto-partitioned by XLA under pure FSDP (`TP=1`). A manual `shard_map` rewrite of the loss calculation is required, which violates our codebase constraints. This represents a hard wall for FSDP on Qwen3-8B. The prior `v036` baseline (`34.6%` MFU) remains the architectural ceiling for this lane.
