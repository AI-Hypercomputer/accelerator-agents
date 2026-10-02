---
title: "qwen3-jax-v057-d-tokamax-layer-ckpt"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Applying jax.checkpoint per-layer (Qwen3DecoderLayer) instead of to the entire loss_fn will fix the 327 GB HBM OOM caused by the compiler auto-rematting all 28 layers simultaneously during the Tokamax CE backward pass."
status: filed
created: 2026-06-15
origin: retrospective-recommendation
---

# qwen3-jax-v057-d-tokamax-layer-ckpt

## Hypothesis under test

**Hypothesis**: In `v057-c`, we hit a massive 327.4 GB HBM OOM because we removed `jax.checkpoint(loss_fn)` entirely without adding per-layer checkpointing. This caused XLA's automatic rematerialization to schedule the recomputation of all 28 layers' `QKV` / `MLP` projections simultaneously for the backward pass of Splash attention and Tokamax CE.
By properly applying `jax.checkpoint` to `Qwen3DecoderLayer.__call__` and NOT wrapping `loss_fn`, we save layer inputs and recompute intra-layer activations one layer at a time. This should fit within the 31.25 GB HBM.

**Mechanism**:
1. Added `jax.checkpoint` wrapping around `Qwen3DecoderLayer._call_impl`.
2. Removed `jax.checkpoint(loss_fn)` from `train.py`.
3. Continuing to use `use_tokamax_ce=True` to avoid the 39.8 GB OOM caused by logits materialization.

**Predicted signal**: The model should successfully compile and train at `batch_size=8`, `seqlen=8192` with Tokamax CE and Splash Attention without running out of HBM or VMEM.

**Falsification criterion**: If the compiler still hits an HBM OOM > 31.25 GB, or if the runtime MFU is poor due to excessive communication.

## Setup
`tmp_upload/jax_v057_c/trainer` contains the code. Building `v6e8-qwen3-8b-jax-20260615-exp057-d` and running via XPK.

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
