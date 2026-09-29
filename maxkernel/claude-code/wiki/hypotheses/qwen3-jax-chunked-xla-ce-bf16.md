---
title: "Qwen3 jax — chunked_xla CE with bf16 inputs"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: invalid
expected_gain: "vocab-head transient shrinks to chunk size (memory unlock for bs2/seq8192); step time ≤ plain CE"
confidence: medium
effort: S
origin: 2026-06-12-v006-tkmce-bf16
hlo_prefilter: passed
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — chunked_xla CE with bf16 inputs

*Hypothesis*: `tokamax.linear_softmax_cross_entropy_loss(implementation=
"chunked_xla")` with bf16 inputs replaces the monolithic `[BL,V]` logits
materialization with chunk-sized transients, recovering the vocab-head memory
(2.32 GB f32 + grads at bs2) without the f32-cast overhead that refuted
[v003](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v003-tkmce.md)
and without the mosaic_tpu bf16-Ref crash that invalidated
[v006](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v006-tkmce-bf16.md).

*Mechanism*: chunked_xla is pure XLA (no dtype-restricted Pallas refs).
Cross-model precedent: the llama3 torchax lane's accepted frontier recipe was
chunked_xla + bf16-native + autotune (its exp 62b/66/66b: chunked_xla beat
mosaic_tpu by +1.6%, bf16-native added +0.3%). Caveat from the generic index:
chunked_xla's *total HBM traffic* is higher than a streamed Pallas kernel —
the win claimed here is transient-memory + step time vs plain CE, judged by
measurement, not FLOPs reasoning.

*Falsification criterion*: loss diverges from plain-CE reference beyond bf16
noise (invalid), or step time regresses vs the 453 ms splash frontier, or
peak/program HBM not lower than plain CE → refuted.

*HLO pre-filter*: passed — inherited (XLA materializes 1.86 GB logits buffers
at bs1; see [qwen3-jax-tokamax-ce](qwen3-jax-tokamax-ce.md)).

## See also

- [v006 (invalid — mosaic_tpu bf16-Ref crash)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v006-tkmce-bf16.md)
- [v003 (refuted — f32-cast overhead)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v003-tkmce.md)

> [!warning] Invalidated by [v008b](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v008-chunked-xla-ce.md) on 2026-06-12
> chunked_xla accumulates in `x.dtype`: bf16-x ⇒ bf16 lse ⇒ −0.73 ≈ ln 2 loss
> bias. Continues as [f32-x-only variant](qwen3-jax-chunked-xla-ce-f32x.md) (v011).
