---
title: "v038: TP=1 Vocab-Parallel CE (bs=8) - RoPE/Norm Fusion"
date: "2026-06-04"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v038 - TP=1 Vocab-Parallel CE (bs=8) - RoPE/Norm Fusion

## Hypothesis under test
- **Hypothesis**: In `v027`, we reached 32.8% MFU with `bs=8` FSDP `TP=1` using Splash attention and native XLA cross-entropy loss. The profile showed a 6.9% `data formatting` bucket consisting of thousands of copy operations. This is caused by XLA being unable to fuse the QK-norm (RMSNorm) and RoPE operations due to the intervening `jnp.transpose` operation from `(B, T, H, Dh)` to `(B, H, T, Dh)`. By applying RoPE *before* the transpose (i.e. on the `[B, T, H, Dh]` tensor), XLA's fusion engine can compile the projection, QK-norm, and RoPE into a single tight loop, eliminating the intermediate memory materialization and the copy bucket, pushing MFU past our 35% goal.
- **Mechanism**: In `model/modeling_qwen3.py`, move `jnp.transpose` after `apply_rotary_pos_emb`, and change `unsqueeze_dim` from 1 to 2 so the rotary embeddings broadcast across the heads dimension. Use standard `xla` Tokamax implementation.
- **Predicted signal**: The `data formatting` bucket shrinks or disappears, and MFU hits >35%.
- **Falsification criterion**: MFU does not improve.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 8
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash
  - Tokamax `implementation="xla"`
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Vocab-Parallel FSDP CE
  - Pre-Transpose RoPE (Fusion optimization)

## Setup Steps
1. Fork codebase from `v026`/`v027`.
2. Edit `model/modeling_qwen3.py` to move the transpositions of `q, k, v` to after RoPE application, updating `unsqueeze_dim=2`.
3. Launch via `gke-cluster-runner` with `--batch_size=8 --tp_parallelism=1 --use_splash=True`.

## Results
- **Status**: Failed (Crash)
- **Throughput**: N/A
- **Approx MFU**: N/A

## Verdict
Falsified (Crash). Fusing RoPE *before* transpose pushes XLA's compiler limits over the edge on the Splash Pallas kernel. While it mathematically computes correctly, altering the layout changes the memory boundaries fed into Splash, causing the backward pass (`splash_mha_dkv_no_residuals`) to attempt to allocate 34.90M on the vmem stack (which has a hard limit of 32.00M), crashing during step compilation with `RESOURCE_EXHAUSTED`.
We cannot utilize pre-transpose RoPE without either modifying Splash's layout assumptions or altering XLA's heuristic choices.
