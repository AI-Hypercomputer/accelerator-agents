---
title: "qwen3-jax: port the certified seq-8192 frontier levers to seq-2048"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "5-15% tok/s/chip at seq2048"
confidence: medium
effort: S
origin: human
tags: [qwen3-cc5, jax, v6e-8, seq2048, flags, weight-layout, charles-v6e]
created: 2026-06-12
updated: 2026-06-12
---

# Port the seq-8192 frontier levers to seq-2048

The seq-2048 best (31.4% MFU,
[v007](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v007-bs2-vmem.md))
predates every frontier lever landed during the seq-8192 campaign. Each is
launch-flag/env-only on the current image and individually attributable.

## Statement

At bs2/seq2048, adding (a) the 21-flag LIBTPU stack and (b)
`JAX_WEIGHT_LAYOUT=io` lifts tok/s/chip ≥ 5% over the v007 recipe
(>6,400/chip vs 6,095), with loss parity.

## Rationale

- **21-flag stack** (+10.3% at seq8192, [v026](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v026-mt-flags-bs3.md)):
  untested at 2k. The v007 profile shows **~30% collectives** (CP-done 15.7% +
  AG 14.3%) — exactly the bucket the RS-layout/windowed-einsum/scheduler flags
  attacked at 8k. Highest-confidence transfer.
- **io weight layout** (+2.8% at seq8192, [v050](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v050-weight-io.md)/v051):
  transpose-free x@W is shape-independent in principle; conv fusion is 37.7%
  of the v007 step.
- NOT in scope (priors say net-cost at 2k, revisit only after the above):
  chunked CE (+3.2% cost, v003-era), remat (+27% cost), context checkpointing
  (only matters under remat), scan (step-neutral substrate; adopt for compile
  speed if flat).

## Proposed experiment

v053 (after v052 continuity PASS): v007 recipe + 21-flag stack (single axis).
v054: + `JAX_WEIGHT_LAYOUT=io`. Each 20 steps, profile 12–14, HLO; compare
against v052's measured baseline on the same cluster. Then batch-rescale
(bs3/bs4 — the v026 flags freed ~8 GiB at 8k; v007 had only 3.1 GiB headroom
at bs2) as the follow-on.

## Risks

None semantic (flags + layout are numerics-preserving; v026/v050 verified
loss parity). VMEM flag value differs between stacks (81920 in v007 vs 98304
in the 21-flag stack — the stack's value supersedes; both > the 33.84 MB dkv
tile need).

## Dependencies

[v052 continuity](qwen3-jax-2k-cluster-continuity.md) must PASS first.

## See also

- [Model page](../models/qwen3-cc5-jax.md) · [Phase summary](../analyses/2026-06-12-maxtext-gap-closing-phase-summary.md)
