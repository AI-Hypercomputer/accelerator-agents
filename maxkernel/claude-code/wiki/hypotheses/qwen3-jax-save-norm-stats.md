---
title: "Qwen3-jax: save norm rsqrt stats on-device (skip f32 reduction recompute)"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "SUPPORTED (marginal) — v045 +0.63% / −26ms / +0.2pp MFU; on-device save skips f32 norm-reduction recompute"
confidence: low
effort: S
origin: human
hlo_prefilter: n/a
tags: [qwen3-cc, jax, remat, rmsnorm, qk-norm, save-only-these-names, on-device, seq8192]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3-jax: save norm rsqrt stats on-device

## Statement

Tagging the RMSNorm/QK-norm rsqrt reduction statistic with `checkpoint_name("norm_rsqrt")` and selecting a
`save_only_these_names("norm_rsqrt")` remat policy keeps the tiny `(B,T,1)`/`(B,T,H,1)` stats ON-DEVICE (not
host → sidesteps the offload-pipeline wall that refuted v039), letting the backward skip recomputing the f32
mean-square reduction (xprof: QK-norm reduce remat `fusion.671` 685ms + RMSNorm `add_rsqrt_fusion`), lifting the
frontier modestly above v035.

## Rationale

The [v035 profile](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md)
deep-dive measured the QK-norm reduce remat at 685ms and the RMSNorm rsqrt chains as separate f32 fusions. The
stats are trivially small (fits the 2.92GB free HBM many times over). Value-preserving (saving an intermediate
never changes values → equivalence holds by construction). The dominant SiLU recompute (21.7GB) can NOT be
saved on-device, so this only addresses the norm-reduction slice.

## Proposed experiment

- v045: image `qwen3-8b-jax:v045-save-norm-stats` + env `SAVE_NORM_STATS=1`, v035 flags. Support: > 6,030 beyond
  noise. Refute: ≤ 6,030 (XLA already fuses the reduction with the elementwise → saving the stat is a no-op).

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md) · [v035](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md)
