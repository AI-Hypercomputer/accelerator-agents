---
title: "Qwen3 jax — Splash VMEM + Tokamax CE Fixed (Retry bs=8, seqlen=8k)"
type: experiment
hypothesis: "The `v056` run crashed with a 40GB HBM OOM. Investigation revealed that the `jax.checkpoint` policy `checkpoint_dots_with_no_batch_dims` causes the outer checkpoint to recompute the Pallas kernel forward pass. Removing this policy avoids the logits materialization and should allow Tokamax CE + Splash to compile and run successfully within HBM."
model: qwen3-cc-jax
variant: 8B/v6e-8
commit: ""
status: refuted
tags: [qwen3, jax, tokamax, splash, vmem, seq8k]
created: 2026-06-15
updated: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-tokamax-fix
---

# Splash VMEM tuning + Tokamax CE checkpoint policy fix (Retry)
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

**Status**: FAILED (HBM OOM)
The compilation failed again with an almost identical 39.8GB HBM allocation request (`RESOURCE_EXHAUSTED: Allocation (size=39829110784) would exceed memory (size=33822867456)`).

Analyzing the HLO dump from the failed run confirms that `size=39829110784` matches exactly the byte size for the `f32` logits materialized matrix:
`batch(64) * seq(8192) * vocab(151936) * 4 bytes = 39,829,110,784 bytes`

However, upon inspecting the code that was executed, it was discovered that `use_tokamax_ce` flag had been inadvertently dropped from `tmp_upload/jax/trainer/train.py` during previous environment debugging. Thus, `v057-b` actually executed the standard JAX cross-entropy (which materializes logits) rather than the streaming Tokamax CE implementation.

## Verdict
**Verdict**: refuted (due to missing Tokamax implementation)
The run OOMed exactly where expected without Tokamax CE. We must retry the experiment with Tokamax CE properly included and enabled.
