---
title: "Qwen3 jax — splash-aware remat (save_attn policy)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "frees FFN-intermediate activation mass (≈60% of the ~25 GB at seq8192) without re-entering splash — unlocks seq-8192 and possibly bs3"
confidence: medium
effort: M
origin: 2026-06-12-v012-remat
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — splash-aware remat (save_attn policy)

*Hypothesis*: Per-layer `jax.checkpoint` with
`save_only_these_names("attn_q","attn_k","attn_v","attn_out")` rematerializes
norms + FFN intermediates while never re-entering the splash custom_vjp —
avoiding [v012](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v012-remat.md)'s
NaN + 27% overhead — and frees enough activation memory for seq-8192 (and
possibly bs3) at single-digit % overhead.

*Mechanism*: `checkpoint_name` tags on q/k/v and the kernel output; the policy
saves exactly those. FFN gate/up intermediates (`[B,T,12288]`, the measured
activation hog in v012-ph2/v013/v014) are recomputed in backward (2 matmuls +
silu). CPU smoke: remat-vs-none loss bit-identical.

*Falsification criterion*: NaN again (→ remat-with-splash deeper issue), or
bs2 overhead > 10%, or seq-8192 still OOMs → refuted for the respective goal.

## See also

- [v012 (invalid — NaN with dots-saveable policy)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v012-remat.md)
- [Observation: remat-splash NaN](../observations/qwen3-jax-remat-splash-nan.md)
- [seq-8192 target](qwen3-jax-seq-8192.md)

**Supported** by [v015](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v015-save-attn-remat.md)
(2026-06-12): no NaN, seq-8192 fits (27.62 GiB), 31.8% MFU at the target
shape. Overhead +27% at bs2/seq2048 → per-shape knob: ON at seq 8192 only.
Merged to trunk a2f1914.
