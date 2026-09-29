---
title: "v035: TP=1 Vocab-Parallel CE (bs=8) - XLA SDPA"
date: "2026-06-04"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v035 - TP=1 Vocab-Parallel CE (bs=8) - XLA SDPA

## Hypothesis under test
- **Hypothesis**: In `v027`, we reached 32.8% MFU with FSDP `TP=1` (`bs=8`) using Splash attention. The profile showed 12.3% of step time stuck in `collective-permute-done` (likely due to Splash's Ring Attention) and 11.2% in `custom-call` (Splash kernel). Since `TP=1` means there is no tensor or sequence parallelism across chips, Splash's sequence-parallel communication is unnecessary and harmful. By reverting to native XLA SDPA (`use_splash=False`), the compiler can natively run its highly optimized FlashAttention-equivalent on the local 2048 sequence length without triggering sequence-parallel communication. This should eliminate the 12.3% communication overhead and push MFU past the 35% goal.
- **Mechanism**: Use the `v027` codebase (which removed the outer checkpoint wrapper to fix fragmentation) but omit the `--use_splash=True` flag to fall back to `jax.nn.dot_product_attention` via XLA.
- **Predicted signal**: `collective-permute-done` drops significantly, and MFU jumps from 32.8% to >35.0%.
- **Falsification criterion**: XLA SDPA OOMs or MFU does not improve.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 8
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - XLA SDPA (Splash disabled)
  - Tokamax `implementation="xla"`
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Re-use the codebase from `v027`.
2. Launch via `gke-cluster-runner` with `--batch_size=8 --tp_parallelism=1` (omitting `--use_splash=True`).

## Results
- **Status**: Failed (OOM during compilation)
- **Throughput**: N/A
- **Approx MFU**: N/A

## Verdict
**Refuted**. Replacing Splash with XLA SDPA at `bs=8` caused the graph to exceed the 31.25GB HBM limit during compilation (Requested 35.67GB, exceeded by 4.42G). This means that Splash attention actually uses *less* memory than XLA SDPA for our specific sequence length and model architecture, keeping us under the 31.25GB limit in `v027`. We cannot fallback to XLA SDPA without dropping the batch size.
