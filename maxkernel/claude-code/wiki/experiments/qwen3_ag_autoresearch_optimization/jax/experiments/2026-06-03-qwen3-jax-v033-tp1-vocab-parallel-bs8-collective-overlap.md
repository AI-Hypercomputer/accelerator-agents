---
title: "v033: TP=1 Vocab-Parallel CE (bs=8) - Collective Overlap Bundle"
date: "2026-06-03"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v033 - TP=1 Vocab-Parallel CE (bs=8) - Collective Overlap Bundle

## Hypothesis under test
- **Hypothesis**: In `v031` and `v032`, we learned that alternative Tokamax CE implementations (`chunked_xla` and `mosaic_tpu`) cannot be used to optimize the FSDP all-gather communication because they either don't exist in the installed wheel or do not support automatic SPMD partitioning. However, in `v027`, we reached 32.8% MFU using the default `xla` loss implementation, but the profile showed 12.3% of step time stuck in `collective-permute-done`. We can overlap this communication at the XLA compiler level by providing an aggressive bundle of experimental collective overlap and latency hiding flags (which we previously compiled in `v019`).
- **Mechanism**: Use the `v027` Docker image, but add the phase-2 collective overlap flags to `LIBTPU_INIT_ARGS` and `XLA_FLAGS`.
- **Predicted signal**: The workload compiles and steady-state MFU jumps from 32.8% to >35.0%.
- **Falsification criterion**: The flags cause a compiler hang, crash, or step time does not improve.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 8
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash
  - Tokamax `implementation="xla"`
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Fused RoPE
  - Vocab-Parallel FSDP CE
  - Aggressive XLA Latency Hiding / Collective Overlap flags

## Setup Steps
1. Re-using the `v027` Docker image.
2. Launched via `gke-cluster-runner` with modified `launch_cmd`.

## Results
- **Status**: Failed (OOM)
- **Throughput**: N/A
- **Approx MFU**: N/A

## Verdict
**Refuted**. The workload failed with an HBM OOM (`Used 33.40G of 31.25G hbm`). The aggressive collective overlap flags cause the compiler to keep more tensors alive concurrently to facilitate the communication overlap, which increases peak memory pressure. Because `bs=8` already pushed us to the very edge of HBM capacity, this slight increase from the latency-hiding heuristics pushed us over the limit. This means we cannot use compiler-level heuristics to overlap the communication overhead without dropping the batch size.
