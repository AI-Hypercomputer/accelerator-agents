---
title: "Qwen3 jax — splash block re-tune at seq 8192 (bkv 2048)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "+0.5-1.5% step (splash = 23.7% of step at seq 8192)"
confidence: medium
effort: S
origin: 2026-06-12-v015-save-attn-remat
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — splash block re-tune at seq 8192

*Hypothesis*: `SPLASH_BKV=2048` + `SPLASH_BKV_COMPUTE=2048` (fwd kv block
1024→2048) reduces the splash share (fwd 13.0% + dkv 10.7% of step at
seq 8192, [v015 profile](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v015-save-attn-remat.md))
— llama3-jax measured bkv=2048 as the seq-8192 match to MaxText (+0.7%) on
the same hardware.

*Mechanism*: env-only knobs already wired in `splash_attn.py`. VMEM: the
80 MB scoped limit accommodates the larger fwd kv block (bkv_dkv is already
2048).

*Falsification criterion*: tok/s ≤ 44,361 + noise (~0.5%) at the identical
v015 phase-2 shape → refuted (1024 stays optimal on this stack).

## See also

- [v015 (frontier; profile source)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v015-save-attn-remat.md)

> [!warning] Refuted by [v016](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v016-splash-bkv2048.md) on 2026-06-12
> bkv=2048 flat (−0.2% tok/s, within noise) at seq 8192. bkv=1024 stays.
