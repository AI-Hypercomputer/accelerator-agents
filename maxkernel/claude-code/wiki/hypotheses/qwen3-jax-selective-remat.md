---
title: "Qwen3 jax — Selective rematerialization"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "Memory reduction to enable batch scaling to bs=2 (unlocking ~1.5-2.5x throughput)"
confidence: high
effort: S
origin: 2026-06-02-qwen3-jax-v001-batch-scaling
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — Selective rematerialization

*Hypothesis*: Wrapping the loss function in selective rematerialization using `checkpoint_dots_with_no_batch_dims` policy will significantly reduce peak HBM activation memory, enabling the scaling of the per-chip batch size from 1 to 2 without triggering HBM Out-of-Memory.

*Mechanism*: By applying `jax.checkpoint` with the selective policy `checkpoint_dots_with_no_batch_dims` to the loss computation, JAX will store expensive intermediate matrix multiplication outputs (dots) while discarding and recomputing cheap elementwise operations (norms, residuals, activations). This avoids the heavy performance overhead of full rematerialization while reclaiming substantial HBM headroom.

*Falsification criterion*: Wrapping the loss function in selective rematerialization either does not prevent HBM OOM at batch size 2, or introduces a step-time overhead at batch size 1 that exceeds 15% → refuted.
