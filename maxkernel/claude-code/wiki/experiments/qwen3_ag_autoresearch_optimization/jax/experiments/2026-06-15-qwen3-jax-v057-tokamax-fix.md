---
title: "Qwen3 jax — Splash VMEM + Tokamax CE Fixed (bs=8, seqlen=8k)"
type: experiment
hypothesis: "The `v056` run crashed with a 40GB HBM OOM. Investigation revealed that the `jax.checkpoint` policy `checkpoint_dots_with_no_batch_dims` causes the outer checkpoint to recompute the Pallas kernel forward pass, which then falls back to JAX operations or incorrectly traces the logits `[B*L, V]` materialization inside the custom VJP logic. Removing this policy avoids the logits materialization and should allow Tokamax CE + Splash to compile and run successfully within HBM."
model: qwen3-cc-jax
variant: 8B/v6e-8
commit: ""
status: running
tags: [qwen3, jax, tokamax, splash, vmem, seq8k]
created: 2026-06-15
updated: 2026-06-15
---

# Splash VMEM tuning + Tokamax CE checkpoint policy fix
**Date**: 2026-06-15

## Hypothesis under test
**Hypothesis**:
1. The `v056` experiment failed with a 40GB HBM allocation request.
2. The outer `jax.checkpoint` applied to `loss_fn` had the policy `checkpoint_dots_with_no_batch_dims`. This policy interferes with Tokamax CE's internal selective rematerialization of the logits matrix, causing the full `[65536, 151936]` matrix to be materialized.
3. Removing this policy entirely (falling back to default full-save checkpointing at the `loss_fn` boundary, or letting inner `nnx.remat` handle it) resolves the HBM OOM.

**Mechanism**:
- `loss_fn = jax.checkpoint(loss_fn)` instead of `loss_fn = jax.checkpoint(loss_fn, policy=...)`
- Keep `--use_tokamax_ce True` and Splash VMEM tunings.

**Predicted signal**: The compilation will succeed and the workload will execute.

**Falsification criterion**: Compilation still fails with HBM OOM.

## Setup
* Cluster: `<your-cluster>`
* Same docker image as `v047` baseline, but injecting custom `train.py` from `qwen3-jax-v057-tokamax-fix`.
* Batch size: 8

## Results
(To be updated)
