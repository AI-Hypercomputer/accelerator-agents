---
title: "v029: TP=1 Vocab-Parallel CE (bs=8) - Fix Splash Layout Mismatch"
date: "2026-06-03"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v029 - TP=1 Vocab-Parallel CE (bs=8) - Fix Splash Layout Mismatch

## Hypothesis under test
- **Hypothesis**: The `v027` profile revealed a `data formatting` bucket taking 6.9% (~76 ms) of the step time, comprised of thousands of `copy` operations. This is caused by a layout bridge mismatch between the output of Splash attention and the surrounding `Qwen3Attention` module. Splash internally returns `(B, Hq, T, Dh)`, but the rest of the model expects `(B, T, Hq, Dh)` to match XLA SDPA's contract. By explicitly adding a `jnp.transpose` immediately after the Splash call, XLA can fuse the transpose into surrounding operations instead of physicalizing the reshape with thousands of copies, saving ~2.5% MFU and pushing us past our 35% goal.
- **Mechanism**: In `modeling_qwen3.py`'s `_attn_splash` function, explicitly transpose the output of `splash_attn.tpu_splash_attention` from `(0, 1, 2, 3)` to `(0, 2, 1, 3)` before returning it to `Qwen3Attention.__call__`.
- **Predicted signal**: The `data formatting` bucket in the profile shrinks dramatically, and MFU jumps from 32.8% to >35%.
- **Falsification criterion**: MFU does not improve, or the model crashes.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 8
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash (with explicit output transpose)
  - Tokamax `implementation="xla"`
  - Inner `nnx.remat` (outer `jax.checkpoint` removed)
  - Fused RoPE
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Forked `v027` into `v029`.
2. Extracted `modeling_qwen3.py` and modified `_attn_splash` to include `return jnp.transpose(out, (0, 2, 1, 3))`.
3. Re-built the Docker image and launched via `gke-cluster-runner`.

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 50864 tok/s (6358/chip)
- **Approx MFU**: 32.6%

## Verdict
**Refuted**. The explicit transpose did not improve MFU; it slightly regressed from 32.8% to 32.6%. This indicates that the `data formatting` copy overhead is physically unavoidable when converting from Splash's `(B, H, T, Dh)` native layout to the projection layout XLA requires. The transpose in Python merely explicitly instructs XLA to do what it was already doing implicitly during the reshape. We must find another avenue to recover the remaining 2.4% MFU.
