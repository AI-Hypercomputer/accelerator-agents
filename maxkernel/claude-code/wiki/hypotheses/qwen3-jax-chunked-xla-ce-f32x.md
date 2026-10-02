---
title: "Qwen3 jax — chunked_xla CE, f32-x-only boundary"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "loss parity restored at +32 MB; enables the seq-8192 CE memory case (~10 GiB logits at L=8192)"
confidence: high
effort: S
origin: 2026-06-12-v008-chunked-xla-ce
hlo_prefilter: passed
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — chunked_xla CE, f32-x-only boundary

*Hypothesis*: Casting only `x` (hidden, 32 MB at bs1) to f32 before
`chunked_xla` fixes the bf16-lse bias measured in
[v008b](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v008-chunked-xla-ce.md)
(loss −0.73 ≈ ln 2, 1/16-quantized) — because the kernel's accumulation dtype
follows `x.dtype` — while keeping w bf16 (the 2.49 GB f32-w cast refuted in
[v003](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v003-tkmce.md)).

*Mechanism*: `tokamax chunked_xla.py`: `dtype = x.dtype`; lse/accumulators in
that dtype. f32 x ⇒ f32 lse ⇒ reference-parity loss, with chunk-sized vocab
transients. Step-time expectation ≈ v008b's +3.3% vs plain CE at bs1; the
payoff shape is seq 8192 (logits ∝ L vs the fixed 2.49 GB f32 dw accumulator).

*Falsification criterion*: loss still offset beyond bf16 noise vs plain-CE
reference → invalid; or step time > plain-CE + 5% at bs1 → refuted for the
seq-8192 enablement path.

*HLO pre-filter*: passed (inherited; XLA materializes logits buffers — see
[qwen3-jax-tokamax-ce](qwen3-jax-tokamax-ce.md)).

## See also

- [v008b (invalid — bf16 lse)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v008-chunked-xla-ce.md)
- [seq-8192 target](qwen3-jax-seq-8192.md)

**Supported** by [v011](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v011-cxla-f32x.md)
(2026-06-12): parity ≤ 0.007, +3.2% step at seq2048/bs1. Deployed at seq-8192
([v013](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v013-seq8k.md)).
