---
title: "Qwen3 jax — explicit activation sharding constraints at sub-block boundaries"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "0–3% (constrains XLA layout/fusion choices in the conv-dominant step; MaxText pins activations at every sub-block)"
confidence: low-medium
effort: S
origin: 2026-06-12-maxtext-feature-gap-inventory
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — explicit activation shardings

*Hypothesis*: pinning the hidden `[B,T,D]` activations to `P("fsdp",None,
None)` at decoder-layer + attention/MLP-output boundaries (MaxText
`models/qwen3.py:1152-1236` pattern; env-gated `JAX_ACT_SHARDING=1`)
constrains XLA's layout choices enough for > 6,840 tok/s/chip (frontier
+1%) at the v048 config.

*Falsification criterion*: ≤ 6,840 → refuted; XLA's propagated shardings
were already optimal. Semantics-free by construction (constraint only).

## See also

- [Inventory (origin)](../analyses/2026-06-12-maxtext-feature-gap-inventory.md)
- [v049 (the test)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v049-act-shard.md)
