---
title: "Exp 31 — overlap_compute_collective_tc in isolation (inconclusive, −0.13%)"
type: experiment
tags: [experiment, gemma4, xla-flags, collective-overlap, culprit-isolation]
hypothesis: overlap-compute-collective-tc-alone
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "branch perfautoresearch/v6e4-20260423-exp31-overlap-compute-collective-tc"
verdict: inconclusive
---

Enabled only `--xla_tpu_overlap_compute_collective_tc=true` (the correctly-named flag identified in exp 1's follow-up — exp 1 tried `_comms` and got "Unknown"). **Result: 33,330 TPS, 368.67 ms/step steady-state, −0.13 % vs exp 25.** Within noise. Flat.

## Hypothesis

This flag asks the scheduler to overlap tensor-core compute with collective communication. If it's beneficial on this workload, expect +1–3 %. If it's the culprit in the exp 1 bundle regression, expect similar regression. If neither, flat.

## Setup

- Branch `perfautoresearch/v6e4-20260423-exp31-overlap-compute-collective-tc` off trunk at exp 25.
- Zero code changes. `LIBTPU_INIT_ARGS="--xla_tpu_overlap_compute_collective_tc=true"` prepended.

## Results

| Metric | Exp 25 | **Exp 31** | Δ |
|---|---|---|---|
| TPS | 33,372 | **33,330** | **−0.13 %** |
| Step time (mean, steps 2–19) | ~368 ms | **368.67 ms** | +0.18 % |
| Per-step σ | tight | min 367.90 / max 369.20 (n=18) | — |
| Compile step 0 | ~155 s | 176.44 s | +14 % |
| Loss trajectory | 3.82 → 1.55 | 3.83 → 1.84 | match |

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp31-overlap-compute-collective-tc](http://localhost:8791/?run=2026-04-23-gemma4-exp31-overlap-compute-collective-tc) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp31-overlap-compute-collective-tc`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp31-overlap-compute-collective-tc/`](../../../../../raw/profiles/2026-04-23-gemma4-exp31-overlap-compute-collective-tc/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — `overlap_compute_collective_tc=true` alone; flat (−0.13 %).

## Combined conclusion (exp 30 + exp 31)

Both isolated collective-overlap-related flags are no-ops on this workload:
- `latency_hiding_scheduler` (exp 30): flat
- `overlap_compute_collective_tc` (exp 31): flat

Therefore the exp 1 / exp 13 / exp 21 bundle regressions (−12 to −25 % step time) are driven by the `async_collective_fusion*` family, specifically:
- `xla_tpu_enable_async_collective_fusion`
- `xla_tpu_enable_async_collective_fusion_fuse_all_gather`
- `xla_tpu_enable_async_collective_fusion_multiple_steps`

Park these permanently at batch=3 / seq=1024 / fsdp=4. At larger scale (different workload shape with more collectives to hide) they may win; not at this workload.

## Verdict

**INCONCLUSIVE** on primary metric. **INFORMATIVE**: narrows the culprit in the exp 1 bundle to `async_collective_fusion*` only. Not merged.

## See also

- [exp 1 — async-collective-flag bundle (discard)](2026-04-23-exp1-async-collective-flags-rejected.md)
- [exp 30 — latency-hiding-scheduler alone (flat)](2026-04-23-exp30-latency-hiding-solo-potential.md)
- [async-collectives concept](../../../../concepts/async-collectives.md)

## Sources

- `/tmp/gemma4_exp31.log`
- `raw/profiles/2026-04-23-gemma4-exp31-overlap-compute-collective-tc/`
