---
title: "Qwen3 jax — (in,out) weight layout: transpose-free x@W contraction (MaxText DenseGeneral pattern)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "0–8% (high variance): best single candidate for MaxText's compute-bound conv (26.9% BW-util) vs our BW-bound (49%) on identical math"
confidence: low-medium
effort: M
origin: 2026-06-12-maxtext-feature-gap-inventory
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — (in,out) weight layout

*Hypothesis*: storing Linear kernels `(in, out)` and contracting `x@W`
without a transpose (env-gated `JAX_WEIGHT_LAYOUT=io`; sharding specs
auto-reversed so the logical sharding is unchanged; same random draw
transposed at init → seed parity) gives > 6,840 tok/s/chip (frontier +1%)
at the v048 config — the last inventory item and the prime candidate for
the conv-bucket BW-util gap vs MaxText.

*Falsification criterion*: ≤ 6,840 → refuted; the layout complex closes
and the residual −2.6% is documented as MaxText's intrinsic stack edge on
this libtpu.

## See also

- [Inventory (origin + analysis)](../analyses/2026-06-12-maxtext-feature-gap-inventory.md)
- [v050 (the test)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v050-weight-io.md)
