---
title: "v030: TP=1 Vocab-Parallel CE (bs=8) - Chunked XLA CE"
date: "2026-06-03"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v030 - TP=1 Vocab-Parallel CE (bs=8) - Chunked XLA CE

## Hypothesis under test
- **Hypothesis**: In `v024`, the `implementation="chunked_xla"` option for Tokamax CE failed due to OOMing during autotuning, which we later discovered was caused by the 16.87 GB HLO fragmentation from the outer `jax.checkpoint`. With the outer checkpoint removed (fixed in `v026`/`v027`), `chunked_xla` should now successfully compile and run. Because `chunked_xla` overlaps the collective communication of the logits across the `fsdp` dimension better than the standard dense `xla` loss implementation, it should shave off the last remaining 50-100 ms of step time (from the 12.3% `collective-permute-done` bucket), pushing the `bs=8` MFU past 35%.
- **Mechanism**: In `train.py`, set `tokamax.linear_softmax_cross_entropy_loss` to `implementation='chunked_xla'`.
- **Predicted signal**: The workload compiles successfully, the autotuner sweeps configs and finds an optimal block size, and steady-state MFU jumps from 32.8% to >35.0%.
- **Falsification criterion**: `chunked_xla` OOMs again, hangs during autotuning, or step time does not improve.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 8
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash
  - Tokamax `implementation="chunked_xla"`
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Fused RoPE
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Forked `v027` into `v030`.
2. Changed Tokamax implementation to `'chunked_xla'`.
3. Re-built the Docker image and launched via `gke-cluster-runner`.

## Results
- **Status**: Failed (Crash)
- **Throughput**: N/A
- **Approx MFU**: N/A

## Verdict
**Refuted**. The run crashed immediately with `ValueError: Unsupported implementation: chunked_xla`. Although the `tokamax` source repository contains the `chunked_xla` implementation, the `tokamax` wheel installed inside the `torchtitan-images/qwen3-8b-jax` base Docker image is an older version that only supports `xla` and `mosaic_tpu`. Therefore, this specific flag cannot be used. We will pivot to testing `mosaic_tpu` in `v031`.
