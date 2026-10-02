---
title: "Exp 22 — batch=4 with fused_bwd (REJECTED — OOM by 1.25 GiB)"
type: experiment
tags: [experiment, gemma4, oom, batch-growth, hbm-ceiling]
hypothesis: fused-bwd-frees-enough-for-batch4
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: 835137a
verdict: crash
---

> *Backfilled from `RESULTS.tsv` + commit `835137a` message.*

Tried batch=4 on the splash+fused_bwd+bf16 CE stack. **OOM at compile time by ~1.25 GiB — same ceiling pattern as exp 10, 11, 23, and later exp 32.**

## Hypothesis

fused_bwd's memory footprint might be smaller than non-fused bwd (one kernel's scratch vs three). Maybe it buys enough to reach batch=4.

## Result

OOM by the familiar 1.25 GiB. The ceiling is XLA's compile-time planner, not kernel scratch — fused_bwd doesn't change that. batch=4 on 1D fsdp=4 at seq=1024 on v6e-4 is fundamentally blocked by peak-activation accounting, not by the kernel choice.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp22-batch4-fused-bwd](http://localhost:8791/?run=2026-04-23-gemma4-exp22-batch4-fused-bwd) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp22-batch4-fused-bwd`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp22-batch4-fused-bwd/`](../../../../../raw/profiles/2026-04-23-gemma4-exp22-batch4-fused-bwd/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: none (run did not reach training steps)
- **What's inside**: No runtime trace — compile-time HBM OOM by ~1.25 GiB, same ceiling pattern as exp 10, 11, 23. Directory holds crash log + HLO dump up to the OOM stage.

## Verdict

**REJECTED — crash.** Not merged. Batch=4 would require either (a) less activation memory (more aggressive remat with compile-time visibility), (b) tensor-parallel weight sharding (refuted by [exp 32](2026-04-23-exp32-2d-mesh-tp2-rejected.md)), or (c) more HBM (hardware change).

## See also

- [exp 10 — seq=2048 b=2 OOM](2026-04-23-exp10-seq2048-batch2-bf16ce-rejected.md), [exp 11 — offload remat OOM](2026-04-23-exp11-offload-remat-rejected.md), [exp 23 — seq=2048 b=2 with fused_bwd OOM](2026-04-23-exp23-seq2048-batch2-fused-bwd-rejected.md), [exp 32 — 2D mesh OOM at batch=3](2026-04-23-exp32-2d-mesh-tp2-rejected.md).
- [hbm concept](../../../../concepts/hbm.md), [training-memory-budget](../../../../concepts/training-memory-budget.md).

## Sources

- `RESULTS.tsv` row `exp22`.
- Commit `835137a`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp22-batch4-fused-bwd/` — xprof run `2026-04-23-gemma4-exp22-batch4-fused-bwd` at http://localhost:8791/?run=2026-04-23-gemma4-exp22-batch4-fused-bwd

