---
title: "v039 Retrospective: HBM Limits of Batch Scaling"
date: "2026-06-04"
model: "qwen3-8b-jax"
lane: "jax"
---

# Retrospective: v039 (Batch Size Scaling `bs=32`, `seqlen=1024`)

## The Symptom
In experiment `v039`, we attempted to scale the per-chip batch size up to 32 while reducing the sequence length to 1024 to increase arithmetic intensity and cross the 35% MFU threshold. The run failed during the compilation phase with the following error:
`jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.`
`Ran out of memory in memory space hbm. Used 35.80G of 31.25G hbm. Exceeded hbm capacity by 4.55G.`

## Root Cause Analysis
The model's HBM requirements scaled beyond the 31.25 GB capacity of the TPU v6e chip, even with the shortened sequence length.

1. **Memory Components:** The `bs=32` configuration increases the activation memory significantly. Even at `L=1024`, the activations, gradients, and optimizer states for a model of Qwen3's size (8.19B parameters) heavily consume HBM.
2. **TP=1 Limitation:** Because Tensor Parallelism is disabled (`TP=1`), each chip holds its portion of the FSDP sharded parameters and optimizer states. The cross-entropy logits tensor (which has shape `[B * L, V]`, or `[32768, 151936]`) is enormous. While FSDP shards this, the intermediate activations required to compute the backward pass are still large enough to breach the 31.25GB limit.
3. **Sequence Length Amortization:** While reducing sequence length from 2048 to 1024 halves the KV cache and intermediate attention activations, the linear projection activations scale with `B * L`. Pushing `B` to 32 makes `B * L = 32768`, which is identical to `v035` (`B=16, L=2048`), a configuration known to approach or exceed the HBM limit. Here, `Used 35.80G of 31.25G hbm` confirms that this configuration is simply too large for a single v6e chip under `TP=1`.

## Conclusion and Next Directions
Scaling batch size to `bs=32` is unviable under `TP=1` FSDP due to the strict 31.25 GB HBM capacity limit.

### Path Forward: The True Solution
We have exhausted algorithmic optimizations (`v037`, `v038`) and batch scaling limits (`v039`). The only remaining avenue to boost MFU beyond 34.6% (achieved in `v035` at `bs=24, L=2048`) without OOMing or stalling is to implement a mathematically equivalent but computationally more efficient cross-entropy loss calculation: **Tokamax CE**.

The previous attempts to use Tokamax's `chunked_xla` hung due to command queue stalling. However, `v039` proves that we MUST reduce the peak HBM usage of the CE loss to allow either faster dispatch or higher batch sizes. The optimal path is to implement a **custom Pallas CE kernel** (or enable a supported Tokamax Pallas kernel that avoids the `chunked_xla` loops) which computes the loss and gradients in a fused, memory-efficient manner without materializing the full `[B*L, V]` logits matrix in HBM, thereby freeing up memory and reducing the `data formatting` and communication overheads that currently bottleneck the model.
