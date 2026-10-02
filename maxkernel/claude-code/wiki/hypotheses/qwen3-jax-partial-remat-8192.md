---
title: "Qwen3 jax — partial remat at seq 8192 (skip-every-5)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "−2-4% step time (cut ~22% of the remat recompute tax; loop fusion is 14.9% of step)"
confidence: medium
effort: S
origin: 2026-06-12-v016-splash-bkv2048
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — partial remat at seq 8192 (skip-every-5)

*Hypothesis*: Exempting every 5th layer (8 of 36) from save_attn remat spends
~3.5 GiB of the 3.6 GiB HBM headroom at seq 8192 to cut the recompute tax
~22%, lowering step time below v015's 1475.7 ms beyond noise.

*Mechanism*: `JAX_REMAT_SKIP_EVERY=5` (new env knob, v017 fork): layers
`i % 5 == 0` run unchecked (full activations ≈ 0.43 GiB/layer resident);
the rest keep save_attn. CPU smoke passes.

*Falsification criterion*: HBM OOM (headroom estimate wrong), or tok/s ≤
44,361 + 0.5% → refuted (recompute is latency-hidden; headroom better spent
elsewhere).

## See also

- [v015 (target-shape frontier)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v015-save-attn-remat.md)

**Closed** (2026-06-12): supported at skip=5
([v017](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v017-partial-remat.md),
+6.3%), refuted at skip=4
([v018](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v018-prem4.md),
−2.4% — non-monotonic near the HBM ceiling; optimal operating point ≈30.7 GiB).
