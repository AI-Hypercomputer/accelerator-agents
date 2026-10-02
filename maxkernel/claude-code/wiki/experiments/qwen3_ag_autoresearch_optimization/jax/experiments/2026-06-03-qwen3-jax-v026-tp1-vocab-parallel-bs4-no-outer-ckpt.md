---
title: "v026: TP=1 Vocab-Parallel CE (bs=4) - Fix Fragmentation"
date: "2026-06-03"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v026 - TP=1 Vocab-Parallel CE (bs=4) - Fix Fragmentation

## Objective
The objective of this experiment is to fix the massive memory fragmentation (16.87 GB) that caused `v024` and `v021` to OOM during compilation at `bs=4`. The fragmentation was caused by the outer `jax.checkpoint(loss_fn)` with the `checkpoint_dots_with_no_batch_dims` policy, which defeated the inner `nnx.remat` on the unrolled layers, causing XLA to save 36 sets of huge dot product inputs. By removing the outer checkpoint, the inner `remat` correctly frees the intermediate tensors, reducing HLO temp memory significantly and eliminating the fragmentation, thus allowing `bs=4` to fit in memory.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 4
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash
  - Tokamax `implementation="xla"`
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Fused RoPE
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Forked the `v024-tp1-xla-bs4` setup to `wiki/experiments/qwen3_ag_autoresearch_optimization/jax/.repo/2026-06-03-v026-tp1-vocab-parallel-bs4-no-outer-ckpt/`.
2. Extracted `train.py` from the `v024` Docker image.
3. In `train.py`, removed the outer `jax.checkpoint` wrapping of `loss_fn`.
4. Built and pushed the Docker image as `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260603-v026-tp1-vocab-parallel-bs4-no-outer-ckpt`.

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 50015 tok/s (6252/chip)
- **Approx MFU**: 32.2%

## Verdict
**Confirmed**. Removing the outer `jax.checkpoint` successfully mitigated the 16.87 GB HLO temp memory fragmentation by allowing the inner `nnx.remat` on the 36 decoder layers to correctly free memory sequentially. The `bs=4` workload fit into the 31.25 GB HBM and ran successfully, achieving 32.2% MFU (up from 24.3% at `bs=2`). TP communication overhead is bypassed. Next, we will scale to `bs=8` (v027) to push MFU over the 35% goal.
