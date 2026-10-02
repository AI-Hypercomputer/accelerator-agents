---
title: "v027: TP=1 Vocab-Parallel CE (bs=8) - Fix Fragmentation"
date: "2026-06-03"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v027 - TP=1 Vocab-Parallel CE (bs=8) - Fix Fragmentation

## Objective
The objective of this experiment is to test whether `bs=8` fits in memory under the `TP=1` FSDP Vocab-Parallel CE configuration, now that the massive memory fragmentation issue has been fixed by removing the outer `jax.checkpoint` wrapper in `train.py` (see `v026`).

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

## Setup Steps
1. Re-uses the codebase and Docker image from `v026`: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260603-v026-tp1-vocab-parallel-bs4-no-outer-ckpt`.
2. Launched via `gke-cluster-runner` with `--batch_size=8`.

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 51064 tok/s (6383/chip)
- **Approx MFU**: 32.8%

## Profile Analysis

**Bucket attribution** (steady state, ignoring step 1-3):

| Bucket | % of step | Top op | Self time |
|---|---|---|---|
| convolution fusion | 47.4% | fusion.12425 (and dups) | 525 ms |
| loop fusion | 17.0% | | 188 ms |
| collective-permute-done | 12.3% | collective-permute-done | 136 ms |
| custom-call | 11.2% | splash_mha_fwd/bwd | 124 ms |
| data formatting | 6.9% | copy.5087 (and dups) | 76 ms |
| custom fusion | 1.7% | | 19 ms |

**Dominant ops** (top 3 by self-time):
1. **convolution fusion** in module `jit_train_step`: 47.4% of step — main matmuls; MXU work.
2. **collective-permute-done** in module `jit_train_step`: 12.3% of step — Ring Attention / P2P communication sync waits.
3. **custom-call** in module `jit_train_step`: 11.2% of step — Splash attention kernel execution.

**Notable patterns**:
- **6.9% `data formatting` bucket**: Sourced entirely from thousands of small `copy` ops (e.g. `copy.5087`). This often indicates a layout bridge mismatch (like QKV-to-Splash) or fragmentation boundaries forcing memory repacking. Resolving this recovers exactly the missing ~2-3% MFU.
- **12.3% `collective-permute-done` bucket**: In a TP=1 run, this typically indicates Splash sequence-parallel (Ring Attention) communication that is not fully overlapped by computation.

## Verdict
**Confirmed**. The workload successfully compiled and ran at `bs=8` without OOMing, confirming the fragmentation fix works for larger batches. The MFU scaled to 32.8% at `bs=8`. We have completely removed TP overhead, but to cross the 35% MFU threshold we likely need to scale the batch size further (e.g. `bs=16`), or run a final profile to find the last bottleneck.
