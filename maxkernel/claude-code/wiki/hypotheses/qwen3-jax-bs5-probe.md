---
title: "Qwen3 jax — bs5 HBM headroom probe at seq8192"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "up to +1% tok/s/chip (diminishing returns band: bs4 delivered +1.8%; bs3→4 amortization mostly linear-with-tokens; sub-linear RS bucket will amortize further)"
confidence: low
effort: S
origin: 2026-06-12-v027-bs4-mtfl
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — bs5 HBM headroom probe at seq8192

*Hypothesis*: bs5 @ seq8192 on the v027 frontier config (scan + offload_attn +
splash + chunked CE f32-x + 25-flag stack) fits in HBM and delivers a
measurable tok/s/chip gain — though diminishing returns suggest the gain is
modest (< +1%) and the main value is confirming headroom before pivoting to
other optimization directions.

*Rationale*: [v027](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v027-bs4-mtfl.md)
confirmed bs4 actual peak = 23.72 GiB (75.9% of 31.25 GiB cap), with a
4.06 GiB/bs slope observed from the bs3 baseline (19.66 GiB) to bs4 (23.72
GiB). Projecting to bs5: 23.72 + 4.06 = **27.78 GiB** (88.9% of cap), leaving
~3.47 GiB free. The profile agent's more conservative estimate is 28.19 GiB
(90.2%), leaving ~3.06 GiB. Either way, fits in theory.

The caution: (a) the async-done (FSDP collective wait) bucket scaled at
1.36x at bs3→bs4 — super-linear, meaning it is growing faster than tokens.
At bs5 it would consume ~13% of step vs 11.5% at bs4. (b) The per-step
amortization from the dominant linear-scaling buckets (conv fusion, splash)
has been halving with each batch increment; bs5 may deliver < +1%.

*Proposed experiment*: launch-only `--batch_size=5` on the v027 config. If OOM
at compile → confirms the slope estimate was accurate and bs5 is infeasible on
this substrate without further HBM reduction. If it runs: measure median step
and tok/s/chip; compare to the 6,040 bar.

*Falsification criterion*: OOM (batch axis ceiling confirmed at bs4), or
tok/s/chip ≤ 6,040 + 1% (batch axis asymptoted; no further bs-scaling value).

*Dependencies*: best run after [v028 splash block sweep](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v028-splash-blocks-bs4.md) — if v028 shifts the frontier, update the baseline step-time reference for bs5's break-even calculation.

## See also

- [v027 (bs4 frontier, HBM data)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v027-bs4-mtfl.md)
- [qwen3-jax-batch-scaling](qwen3-jax-batch-scaling.md)
