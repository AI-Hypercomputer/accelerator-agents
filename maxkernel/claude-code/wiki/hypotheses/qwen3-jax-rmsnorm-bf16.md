---
title: "Qwen3-jax: RMSNorm bf16 weight-multiply (match MaxText rms_norm)"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "PARITY — v042 +0.16% (sub-noise); RMSNorm fp32-mult is a small slice (loop-fusion is SiLU-dominated, already bf16)"
confidence: medium
effort: S
origin: human
hlo_prefilter: n/a
tags: [qwen3-cc, jax, rmsnorm, bf16, loop-fusion, maxtext, seq8192]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3-jax: RMSNorm bf16 weight-multiply

## Statement

Our `Qwen3RMSNorm` keeps fp32 through the weight-scale multiply (`(x32·rsqrt·weight_f32).astype(bf16)`) over
the full `(B,T,D)` tensor; MaxText's `rms_norm` downcasts `(x·rsqrt)` to bf16 **before** the weight scale, so
that elementwise runs in bf16 not fp32 — halving its bytes, paid twice (fwd + recompute under
`nothing_saveable`). Matching MaxText trims the fp32 slice of the loop-fusion bucket, lifting the seq8192
frontier modestly above v035.

## Rationale

The [v035 profile](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md)
deep-dive confirmed the RMSNorm rsqrt/reduction runs in f32 (`add_rsqrt_fusion`, `copy.388` f32 at 86.6% BW).
SiLU (the dominant 3,985ms) is already bf16, so the norm is the only fp32 inefficiency in loop-fusion. Change is
provably fp32-identical (equivalence test runs fp32) and matches the MaxText reference exactly — no semantics
risk. See [normalizations.py](../../?).

## Proposed experiment

- v042: image `qwen3-8b-jax:v042-rmsnorm-bf16` (v033-maxtext-ce + ONLY the RMSNorm change), v035 flags. Support:
  > 6,030 beyond noise. Refute: ≤ 6,030.

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md) · [v035](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md)
