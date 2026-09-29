---
title: "v031: TP=1 Vocab-Parallel CE (bs=8) - Mosaic TPU CE"
date: "2026-06-03"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v031 - TP=1 Vocab-Parallel CE (bs=8) - Mosaic TPU CE

## Hypothesis under test
- **Hypothesis**: The `v030` run testing `chunked_xla` crashed because that implementation is not available in the older `tokamax` wheel installed in the `qwen3` base Docker image. However, the `mosaic_tpu` implementation is available. In `v024`, `mosaic_tpu` failed during autotuning due to an OOM, but we now know that OOM was caused by the 16.87 GB HLO fragmentation from the outer `jax.checkpoint`. With the outer checkpoint removed (fixed in `v026`/`v027`), `mosaic_tpu` should now successfully compile and run. Because `mosaic_tpu` uses a highly optimized Pallas kernel to overlap the collective communication of the logits across the `fsdp` dimension better than the standard dense `xla` loss implementation, it should shave off the last remaining 50-100 ms of step time (from the 12.3% `collective-permute-done` bucket), pushing the `bs=8` MFU past 35%.
- **Mechanism**: In `train.py`, set `tokamax.linear_softmax_cross_entropy_loss` to `implementation='mosaic_tpu'`.
- **Predicted signal**: The workload compiles successfully, the autotuner sweeps configs without OOMing, and steady-state MFU jumps from 32.8% to >35.0%.
- **Falsification criterion**: `mosaic_tpu` OOMs again, hangs during autotuning, or step time does not improve.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 8
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash
  - Tokamax `implementation="mosaic_tpu"`
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Fused RoPE
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Forked `v027` into `v031`.
2. Changed Tokamax implementation to `'mosaic_tpu'`.
3. Re-built the Docker image and launched via `gke-cluster-runner`.

## Results
- **Status**: Failed (Crash)
- **Throughput**: N/A
- **Approx MFU**: N/A

## Verdict
**Refuted**. The workload crashed during compilation inside the `mosaic_tpu` kernel: `ValueError: Invalid dtype for swap. Ref dtype: float32. Value dtype: bfloat16.`. This occurs because the `mosaic_tpu` Pallas kernel expects the dot product of `x` and `w` to yield `float32` into the scratch buffer, but the inputs are `bfloat16` and the matmul seems to produce `bfloat16`. We need to either cast the inputs or find a way to specify the accumulator dtype.
