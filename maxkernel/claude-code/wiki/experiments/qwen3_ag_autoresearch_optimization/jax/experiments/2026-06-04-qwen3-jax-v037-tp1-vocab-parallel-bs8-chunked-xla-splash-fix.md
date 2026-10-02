---
title: "v037: TP=1 Vocab-Parallel CE (bs=8) - Chunked XLA CE (Splash Fix)"
date: "2026-06-04"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v037 - TP=1 Vocab-Parallel CE (bs=8) - Chunked XLA CE (Splash Fix)

## Hypothesis under test
- **Hypothesis**: In `v036`, using `chunked_xla` as the cross-entropy implementation caused a VMEM OOM in the `splash_mha_dkv_no_residuals` kernel (the backward pass of Splash Attention). Because `chunked_xla` partitions the logits across `fsdp`, the layout of the back-propagated gradient `dx` likely changed compared to the pure `xla` CE loss. This layout change forces the `splash_mha_dkv` custom call to allocate an extra layout conversion buffer in SRAM (VMEM), pushing it over the 32MB limit. By explicitly configuring the Splash backward pass to use smaller block sizes (`SPLASH_BQ_DKV=1024` and `SPLASH_BKV_DKV=1024` instead of the default 2048), we will reduce the SRAM footprint of the kernel enough to accommodate the layout conversion buffer and avoid the VMEM OOM, allowing `chunked_xla` to successfully overlap the FSDP all-gather communication.
- **Mechanism**: Use `v036` as the base, but explicitly inject `os.environ["SPLASH_BQ_DKV"] = "1024"`, `os.environ["SPLASH_BKV_DKV"] = "1024"`, and `os.environ["SPLASH_BKV_DKV_COMPUTE"] = "1024"` into `train.py` before `set_splash_mesh()`.
- **Predicted signal**: The experiment compiles without VMEM OOM and successfully runs. The MFU should increase past 32.8% towards 35% as the `collective-permute-done` overhead shrinks.
- **Falsification criterion**: The model continues to OOM in VMEM or HBM, or performance remains < 32.8% MFU.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 8
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash (with reduced backward block sizes)
  - Tokamax `implementation="chunked_xla"` (injected)
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Fused RoPE
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Forked `v036` into `.repo/2026-06-04-v037-tp1-vocab-parallel-bs8-chunked-xla-splash-fix`.
2. Modified `train.py` to set the smaller block sizes for Splash backward pass.
3. Re-built the Docker image and launched via `gke-cluster-runner`.

## Results
- **Status**: Pending
- **Throughput**: TBA
- **Approx MFU**: TBA

## Profile Analysis
TBA

## Verdict
TBA
