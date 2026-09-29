---
title: "Qwen3 jax — fused gate-up MLP projection"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "2–6% step time (targets the 52.2% convolution fusion bucket; MXU gap 55.7% vs ~65%+)"
confidence: medium
effort: M
origin: 2026-06-12-v044-context-checkpoint
hlo_prefilter: passed
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — fused gate-up MLP projection

*Hypothesis*: fusing the gate and up projections into a single grouped
matmul (as MaxText's `fused_kernel=True` does in its MLP module) reduces
HBM round-trips for the weight tiles and may improve MXU utilization from
the current 55.7% toward MaxText's implied ~65%+.

*Rationale*: after v044, the convolution-fusion bucket (52.2%, 1,272 ms)
is the sole dominant cost — the splash custom-call bucket (now 21.7%) and
remat scope (now smaller) have both been addressed. The MXU utilization gap
(55.7% vs MaxText ~65%+) maps to ~85 ms of step time at the current op-point.
MaxText's Qwen3 config uses `fused_kernel=True` in its MLP, which combines
gate+up into a single `[H, 2×F]` weight matrix and makes one collective
instead of two; this may produce a tighter HLO fusion with fewer barriers
and better MXU continuity. The `[H, 2×F]` weight also avoids the per-layer
FSDP gather boundary between gate and up.

*HLO pre-filter*: pending — must AOT-compile the fused-MLP variant and
confirm XLA does not already fuse gate+up into a single kernel. If the
baseline HLO already produces a `fused_computation` containing both
gate_proj and up_proj matmuls as a single Mosaic kernel, retire immediately
with `status: retired; reason: xla-already-fuses`.

*Falsification criterion*: step time ≤ 2,437 ms (6,723 tok/s/chip, the
v044 frontier) after the fused-MLP change → refuted. Loss must match the
v044 reference within ±0.001.

## See also

- [v044 (origin)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)
- [MaxText feature-gap inventory](../analyses/2026-06-12-maxtext-feature-gap-inventory.md)
