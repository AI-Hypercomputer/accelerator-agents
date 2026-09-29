---
title: "v042 — Qwen3 8B jax: scan-over-layers + fix outer remat"
type: experiment
tags: [qwen3, jax, v6e-8, scan, remat]
variant: 8B/v6e-8
hypothesis: "Replacing the unrolled Python `for` loop over layers with `jax.lax.scan` (via `nnx.scan`) will eliminate the massive `while` loop overhead in XLA. Additionally, removing `jax.checkpoint` on the outer `loss_fn` will prevent full forward-pass recomputation, drastically reducing `fusion` time."
commit: TBD
status: in_progress
verdict: TBD
---

# v042 — Qwen3 8B jax: scan-over-layers + fix outer remat

## Hypothesis under test
**Hypothesis**: Replacing the unrolled Python `for` loop over layers with `jax.lax.scan` (via `nnx.scan`) will eliminate the massive `while` loop overhead in XLA. Additionally, removing `jax.checkpoint` on the outer `loss_fn` will prevent full forward-pass recomputation, drastically reducing `fusion` time.
**Mechanism**: Modify `modeling_qwen3.py` to use `nnx.scan` over `self.layers` in `Qwen3Model.__call__`. Modify `train.py` to remove `jax.checkpoint` over the top-level `loss_fn`.
**Predicted signal**: The trace should show `while` loop overhead drop from ~17.4% to <2%, and `fusion` time should decrease by ~33%. Throughput should increase significantly to >35% MFU.
**Falsification criterion**: If MFU is <35% or if XLA compile fails due to `scan` incompatibility with other ops.

## Results
- **Headline metric**: TBD
- **Details**: TBD

## Verdict
- **Reasoning**: TBD
- **Phase 3 hypothesis-firing audit result**: TBD
