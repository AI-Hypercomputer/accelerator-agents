---
title: "v032: TP=1 Vocab-Parallel CE (bs=8) - Mosaic TPU CE (Patched)"
date: "2026-06-03"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v032 - TP=1 Vocab-Parallel CE (bs=8) - Mosaic TPU CE (Patched)

## Hypothesis under test
- **Hypothesis**: In `v031`, the `mosaic_tpu` CE implementation crashed during compilation because the internal Pallas kernel has a bug where it allocates its scratch accumulator as `float32` but attempts to assign it the result of a `bfloat16` dot product (`x_ref @ w_ref`). By explicitly casting `x` and `w` to `jnp.float32` right before the `tokamax.linear_softmax_cross_entropy_loss` call in `train.py`, the inputs to the Pallas kernel will be `float32`. This guarantees that the dot product is naturally `float32`, satisfying the kernel's internal shape constraints. Although saving `w` and `x` in `float32` consumes more memory for the backward pass, our switch to TP=1 FSDP ensures `w` is heavily sharded (only ~308MB per chip), so we have plenty of HBM headroom. With this patch, the Pallas kernel will compile, perfectly overlap the FSDP parameter communication with the blocked logits computation, and shave off the remaining 50-100 ms of step time, pushing MFU > 35%.
- **Mechanism**: In `train.py`, cast `x` and `w` to `jnp.float32` before calling `tokamax.linear_softmax_cross_entropy_loss(..., implementation='mosaic_tpu')`.
- **Predicted signal**: The workload compiles successfully and steady-state MFU jumps from 32.8% to >35.0%.
- **Falsification criterion**: The model OOMs due to `float32` inputs, or the step time does not improve.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 8
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash
  - Tokamax `implementation="mosaic_tpu"` with `jnp.float32` patched inputs
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Fused RoPE
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Forked `v027` into `v032`.
2. Changed Tokamax implementation to `'mosaic_tpu'` and cast inputs to `jnp.float32`.
3. Re-built the Docker image and launched via `gke-cluster-runner`.

## Results
- **Status**: Failed (Crash)
- **Throughput**: N/A
- **Approx MFU**: N/A

## Verdict
**Refuted**. The workload crashed during compilation with `NotImplementedError: Mosaic kernels cannot be automatically partitioned. Please wrap the call in a shard_map.` Because we are using `TP=1` FSDP via `spmd_mesh` (which relies on JAX's automatic SPMD partitioner), we cannot use the `mosaic_tpu` Pallas kernel. Therefore, both `chunked_xla` and `mosaic_tpu` are dead ends for this repository. We must stick to the default `xla` implementation and find another way to optimize communication.
