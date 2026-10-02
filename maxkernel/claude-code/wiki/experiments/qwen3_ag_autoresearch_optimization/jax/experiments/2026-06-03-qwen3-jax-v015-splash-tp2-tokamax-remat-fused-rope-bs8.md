# Experiment: v015 Splash + TP=2 + Tokamax CE + Per-Layer Remat + Fused QK-norm RoPE (bs=8)

**Status:** Setup
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: The Qwen3 architecture applies a per-head RMSNorm (`q_norm` and `k_norm`) before the RoPE positional embedding. Currently, this sequence happens as independent JAX ops. We hypothesize that applying `q_norm` and `k_norm` *after* the transpose to `(B, H, T, Dh)` (making the layout contiguous for the operations) will allow XLA to fuse the Norm and RoPE operations into a single kernel, eliminating element-wise memory bandwidth overhead and reducing loop fusion time.
- **Mechanism**: Use the `v014` image. Edit `modeling_qwen3.py` `Qwen3Attention.__call__` to apply `q_norm` and `k_norm` after the transpose.
- **Predicted signal**: The loop fusion bucket in the profile will shrink, pushing overall MFU higher without changing model semantics.
- **Falsification criterion**: Loop fusion bucket size does not change, or the MFU decreases.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 8
- **Seq Length**: 2048
- **Remat Policy**: `nnx.remat` (per-layer)
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Flags**: `--tp_parallelism=2 --batch_size=8 --use_splash=True`

## Setup
Derived from `v014` (Splash + TP=2 + Tokamax CE + Per-Layer Remat bs=8).
1. In `model/modeling_qwen3.py`, modified `Qwen3Attention.__call__` to apply `q_norm` and `k_norm` *after* the transpose `(0, 2, 1, 3)`, immediately before `apply_rotary_pos_emb`.
2. Built new docker image `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8` containing the modified model code.
