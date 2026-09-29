---
title: "Qwen3 jax — tokamax CE with bf16 inputs"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "−1.9 GB/chip HBM (logits streamed, no f32 cast); step time back to ≤ plain-CE"
confidence: medium
effort: S
origin: 2026-06-12-v003-tkmce
hlo_prefilter: passed
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — tokamax CE with bf16 inputs

*Hypothesis*: Feeding bf16 `hidden`/`lm_head_weight` directly to
`tokamax.linear_softmax_cross_entropy_loss(implementation="mosaic_tpu")` —
no fp32 boundary cast — keeps the logits-streaming benefit (zero `[BL,V]`
materializations) while dropping the 2.49 GB replicated `f32[4096,151936]`
weight cast that made [v003](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v003-tkmce.md)
a net HBM wash and +6.3% step-time regression.

*Mechanism*: The "fp32 boundary cast REQUIRED" rule is `chunked_xla`-specific
(bf16 lse accumulation collapse — llama3 torchax exp 66). `mosaic_tpu`
accumulates inside the kernel; bf16 operands should be safe. Falsified-or-not
by the in-run loss-parity A/B.

*Falsification criterion*: loss collapses/offsets beyond bf16 noise vs plain
CE (kernel-internal accumulation insufficient) → invalid; or HBM/step-time
still not better than plain CE → refuted.

*HLO pre-filter*: passed — inherited from [qwen3-jax-tokamax-ce](qwen3-jax-tokamax-ce.md)
(XLA materializes 1.86 GB of logits buffers at bs1; confirmed again in v003's
module where the kernel removed them).

## See also

- [v003-tkmce (refuted — f32-cast overhead)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v003-tkmce.md)

> [!warning] Invalidated for mosaic_tpu by [v006](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v006-tkmce-bf16.md) on 2026-06-12
> The installed tokamax `mosaic_tpu` backward kernel hardcodes an f32 VMEM
> scratch Ref (`pallas_mosaic_tpu.py:334`) — bf16 inputs crash at Pallas
> trace time. The idea continues as
> [chunked_xla CE bf16](qwen3-jax-chunked-xla-ce-bf16.md) (v008); re-open the
> mosaic_tpu path on a tokamax upgrade with bf16 Ref support.
