---
title: "Qwen3 jax — adopt the 13 residual MaxText recipe flags (windowed-einsum disable + collective fusion + layout)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "+2-8% step at bs3/seq8192 (windowed-einsum disable is the headline; rest are collective overlap/layout)"
confidence: medium
effort: S
origin: 2026-06-12-v025-offload-scan-bs3
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — MaxText recipe flag completion

> **SUPPORTED — FRONTIER ([v026](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v026-mt-flags-bs3.md), 2026-06-12)**:
> +10.3% vs v025 (5,933 tok/s/chip, 34.0% MFU @ bs3/seq8192), numerics
> unchanged. Windowed-einsum disable −341 ms; RS layout −90 ms; HBM peak
> 27.34 → 19.66 GiB; offload DMA stalls 416 → 0.7 ms. Lane-standard.

*Hypothesis*: Adding the 13 MaxText-recipe flags our stack never adopted —
headline `--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`
(disables collective matmul; v025's profile shows the windowed-einsum
collective-permute ring is ACTIVE in our compile), plus the
async-collective-fusion family, reduce-scatter layout flags, DP-overlap
flags, and `vmem_limit_kib` 98304 (vs our 81920) — cuts ≥ 2% from the bs3 @
seq 8192 step (measurable as tok/s/chip > 5,489 = v025 + 2%).

*Mechanism*: MaxText runs the identical shape at 3,535 ms vs our 4,567 ms
with these flags as the only env-level difference besides kernel/remat
choices. Windowed einsum restructures FSDP matmuls into collective-permute
rings that XLA chose for us at bs3; MaxText explicitly turns it off in the
validated recipe. Excluded: the 2 SparseCore flags (compile-fatal on this
libtpu, mt-v001 attempt 1).

*Falsification criterion*: tok/s/chip within noise (±1%) of v025's 5,381 →
refuted; the flag stack is not where the MaxText delta lives.

## See also

- [v025 (the baseline config)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v025-offload-scan-bs3.md)
- [mt-v001 (flag provenance)](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md)
