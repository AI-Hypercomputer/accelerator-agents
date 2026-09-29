---
title: "Qwen3 jax — drop attn_out from the save list (save_qkv) → bs3 fits"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "frees ~6.75 GiB stacked residuals (v039 missed by only 0.6 GiB) → bs3 fits; net tok/s depends on attn_out recompute cost vs batch amortization"
confidence: medium
effort: M (one-line policy variant; grads bit-exact in CPU smoke)
origin: 2026-06-12-v039-bs3-saveattn
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — save_qkv ablation (drop attn_out)

*Hypothesis*: saving only q/k/v (recomputing attn_out in backward — which
already re-runs splash fwd) frees ~6.75 GiB of scan-stacked residuals so
**bs3/seq8192 fits** and beats 6,283 tok/s/chip (frontier +1%); secondary
arm checks save_qkv at bs2 (does dropping attn_out cost anything at the
current frontier shape?).

*Rationale*: [v039](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v039-bs3-saveattn.md)
OOM'd by only 612 MiB with attn_out occupying ~6.75 GiB of the save set;
the 4-name list was inherited from llama3 (v015) and never ablated. CPU
smoke: save_attn vs save_qkv gradients **bit-identical** (recompute is pure
reordering).

*Falsification criterion*: bs3+save_qkv OOMs again, or both arms ≤
frontier+1% → refuted; the save-list ablation closes.

## See also

- [v039 (the 612 MiB wall + allocation table)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v039-bs3-saveattn.md)
- [v041 (the test)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v041-save-qkv-bs3.md)
