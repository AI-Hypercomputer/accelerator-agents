---
title: "v036: TP=1 Vocab-Parallel CE (bs=8) - Chunked XLA CE (Injected)"
date: "2026-06-04"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v036 - TP=1 Vocab-Parallel CE (bs=8) - Chunked XLA CE (Injected)

## Hypothesis under test
- **Hypothesis**: In `v030`, we tried `chunked_xla` but the `tokamax` wheel was too old. In `v023`, we copied the newer `tokamax` from the workspace but the run OOMed due to the outer `jax.checkpoint` issue. Now that we have `bs=8` FSDP=8 TP=1 running successfully (`v027`), but with a 12.3% step-time overhead in `collective-permute-done` (which corresponds to FSDP all-gather of the vocab), we will reinject the newer `tokamax` code and use `chunked_xla`. `chunked_xla` should overlap the `all-gather` of the logits across the `fsdp` dimension with compute, potentially shaving off the 136 ms wait and pushing the MFU from 32.8% to over 35%.
- **Mechanism**: Copy the `tokamax` source directly to `/opt/venv/lib/python3.12/site-packages/tokamax` in the Dockerfile. Use `implementation="chunked_xla"` in `train.py`. Run with `--batch_size=8`, `--tp_parallelism=1`, and Splash.
- **Predicted signal**: The experiment compiles without OOMing and the MFU jumps above 35% as the 12.3% FSDP communication bucket shrinks.
- **Falsification criterion**: The `chunked_xla` implementation crashes, hits an OOM due to increased memory pressure from chunking, or the MFU does not improve.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 8
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash
  - Tokamax `implementation="chunked_xla"` (injected)
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Fused RoPE
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Forked `v035` into `.repo/2026-06-04-v036-tp1-vocab-parallel-bs8-chunked-xla`.
2. Changed `implementation='chunked_xla'` in `train.py`.
3. Added `COPY tokamax /opt/venv/lib/python3.12/site-packages/tokamax` to the Dockerfile (from `v023`'s `.repo`).
4. Re-built the Docker image and launched via `gke-cluster-runner`.

## Results
- **Status**: Pending
- **Throughput**: TBA
- **Approx MFU**: TBA

## Profile Analysis
TBA

## Verdict
TBA
