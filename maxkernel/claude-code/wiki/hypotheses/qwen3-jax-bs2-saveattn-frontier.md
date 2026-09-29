---
title: "Qwen3 jax — bs2 + save_attn + 21-flag stack as the new frontier (the v037 control-arm discovery)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "+3.3% tok/s/chip over v027 (6,240 vs 6,040; 35.8% vs 34.6% MFU) — discovered, needs 50-step validation + profile"
confidence: high
effort: S
origin: 2026-06-12-v037-unroll-saveattn-bs2
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — bs2 + save_attn frontier candidate

*Hypothesis*: bs2/seq8192 + scan + **save_attn** (no offload) + splash +
chunked CE f32-x + the 21-flag stack sustains > 6,200 tok/s/chip over 50
steps with no NaN/drift — making it the lane frontier (replacing
bs4+offload's 6,040).

*Rationale*: discovered as [v037](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v037-unroll-saveattn-bs2.md)'s
control arm (2,625.7 ms, 18 tight steady steps). Mechanism: the no-offload
substrate avoids all host-DMA staging costs and bs2 halves the absolute AG
mass per step; v021's old bs2 figure (5,701) predates save_attn-under-scan
and the v026 flag completion (+9.5% combined at this op-point). All
ingredients are in trunk 174efd7 — launch-only recipe change.

*Falsification criterion*: 50-step run < 6,200 sustained, NaN, or drift →
candidate rejected, v027 stands.

## See also

- [v037 (origin)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v037-unroll-saveattn-bs2.md)
- [v038 (validation)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v038-bs2-saveattn-val.md)
