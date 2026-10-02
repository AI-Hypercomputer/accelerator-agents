---
title: "Qwen3 jax — bs3 + save_attn knife-edge probe on the new frontier substrate"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "+1-2% tok/s/chip if it fits (batch amortization on the no-offload substrate); OOM is decisive data"
confidence: low
effort: S
origin: 2026-06-12-v038-bs2-saveattn-val
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — bs3 + save_attn probe

*Hypothesis*: bs3/seq8192 + save_attn + scan + 21-flag stack fits in
31.25 GiB and beats 6,283 tok/s/chip (v038 frontier +1%).

*Rationale*: [v038's profile](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v038-bs2-saveattn-val.md)
puts bs2 at 28.62 GiB (2.63 free); bs3 projection straddles the cap (−0.27
to +0.48 GiB depending on heap-scaling model). The no-offload substrate
proved more per-token-efficient than offload (v037/v038) — if bs3 fits,
amortization may add a point or two.

*Falsification criterion*: OOM (likely; closes the batch rung with a
measured wall) or ≤ 6,283 → refuted; v038 stands.

## See also

- [v038 (frontier + HBM projection)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v038-bs2-saveattn-val.md)
