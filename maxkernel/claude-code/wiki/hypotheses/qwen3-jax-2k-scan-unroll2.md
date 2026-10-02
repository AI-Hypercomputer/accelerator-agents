---
title: "qwen3-jax: scan unroll=2 at the 2k bs5 frontier — hide the exposed gradient reduce-scatter"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: retired   # refuted v063: unroll=2 = −4.1% regression; scheduler ignores the window; S(5) bug confirmed offload-specific
expected_gain: "3-10% step time at seq2048/bs5"
confidence: medium
effort: S
origin: 2026-06-12-v055-2kstack
tags: [qwen3-cc5, jax, v6e-8, seq2048, scan, unroll, reduce-scatter, charles-v6e]
created: 2026-06-13
updated: 2026-06-13
---

# Scan unroll=2 at 2k — attack the exposed grad-RS

The single largest structural cost at the certified 2k frontier is the
**exposed gradient reduce-scatter: ~158 ms = 12.1% of the bs5 step** —
v052's unrolled graph hid the same work almost completely (ARS bucket
3.3 ms), so the cost is pure scheduling, not bytes. `JAX_SCAN_UNROLL=2`
doubles the scan-body size, giving the latency-hiding scheduler a 2-layer
window to overlap layer N's grad-RS under layer N+1's backward compute.

## Statement

At bs5/seq2048 on the certified v060 recipe, `JAX_SCAN_UNROLL=2` cuts step
time ≥ 3% (≤ 1,262 ms; tok/s/chip ≥ 8,107) with loss parity
(12.0960/12.0969/12.0933 — unroll is graph-shape-only).

## Rationale

- v055/v058 profiles: ARS 146–158 ms exposed under scan; v052 unrolled: 3.3
  ms. Bytes identical — a 2× scheduling window should recover a large
  fraction. Full ceiling = ~150 ms ≈ 11.5%.
- **The XLA S(5) bug does NOT apply**: it fires on unroll≥2 × pinned-host
  offload; the 2k recipe (`save_qkv_ctx_hbm`) keeps everything in HBM.
  First legal unroll≥2 test since the bug was found.
- Memory: residual totals unchanged; some scheduling slack — bs5 sits at
  87.3% (3.97 GiB free), so modest growth tolerable. OOM ⇒ measure margin,
  retry at bs4 (where 7.4 GiB free) to isolate the scheduling effect.
- Compile: body 2× — still tiny (773 KB HLO), well under budget.

## Proposed experiment

v063: v060 recipe + `JAX_SCAN_UNROLL=2`, 20 steps, profile 12–14. Predicted
signal: ARS bucket shrinks toward ≤ 60 ms; scan `while` trip count 18.
If supported, try unroll=3/4 (diminishing; compile grows) and certify.

## Risks

None semantic (scheduling only; loss must be bit-comparable). OOM risk
(scheduling slack) — falls back to bs4 arm.

## Dependencies

Slice free after mt-v002.

## See also

- [v055 decomposition](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v055-2kstack.md) · [v058 profile](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v058-2kbs5.md) · [Model page](../models/qwen3-cc5-jax.md)
