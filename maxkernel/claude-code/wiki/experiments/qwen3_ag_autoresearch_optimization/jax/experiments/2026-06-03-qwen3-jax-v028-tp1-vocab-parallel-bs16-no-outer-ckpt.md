---
title: "v028: TP=1 Vocab-Parallel CE (bs=16)"
date: "2026-06-03"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v028 - TP=1 Vocab-Parallel CE (bs=16)

## Hypothesis under test
- **Hypothesis**: Now that the HLO temp memory fragmentation issue is fixed and we can run `bs=8` under FSDP `TP=1` using Vocab-Parallel CE, we can eliminate the final bound on MFU by scaling batch size to 16. `bs=16` was previously blocked by the 31.25 GB HBM limit (we were at 99.9% at `bs=8` under TP=2). However, by moving to pure FSDP (`TP=1`), the weights are sharded 8-ways instead of 4-ways, which frees up substantial parameter memory, giving us the headroom to scale the batch size.
- **Mechanism**: Re-use the `v026` codebase. Run with `--batch_size=16`.
- **Predicted signal**: The workload will fit in memory (due to 8-way FSDP weight sharding) and MFU will scale past the 35% goal.
- **Falsification criterion**: The workload OOMs during runtime, or MFU does not improve.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 16
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash
  - Tokamax `implementation="xla"`
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Fused RoPE
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Re-uses the codebase and Docker image from `v026`: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260603-v026-tp1-vocab-parallel-bs4-no-outer-ckpt`.
2. Launched via `gke-cluster-runner` with `--batch_size=16`.

## Results
- **Status**: Failed (OOM during compilation)
- **Throughput**: N/A
- **Approx MFU**: N/A

## Verdict
**Refuted**. The workload failed to compile due to a true out-of-memory error (not fragmentation). It requested 35.63GB of HBM, exceeding the 31.25GB capacity by 4.38GB.
The memory breakdown shows:
- Program (HLO Temp): 29.89 GB (very little fragmentation: 1.33G)
- Arguments/Output: 5.72 GB

This indicates that even with 8-way FSDP weight sharding and proper `nnx.remat` usage, the intermediate activations for `bs=16` across the 2048 sequence length and 36 layers are simply too large to fit in a 32GB TPU v6e chip. The 32.8% MFU achieved at `bs=8` (`v027`) is the peak for `TP=1` FSDP without further optimization.

Next steps: We need to look at the `v027` profile to figure out where we can shave off another 2.5% MFU at `bs=8` to hit our 35% goal, since scaling the batch size to 16 is physically impossible without more memory.
