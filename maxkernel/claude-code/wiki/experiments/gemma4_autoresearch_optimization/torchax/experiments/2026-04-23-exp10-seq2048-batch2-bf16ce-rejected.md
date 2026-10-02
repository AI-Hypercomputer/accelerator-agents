---
title: "Exp 10 — seq=2048 batch=2 with bf16 CE (REJECTED — CompileTimeHbmOom)"
type: experiment
tags: [experiment, gemma4, oom, long-seq, batch-growth, hbm-ceiling]
hypothesis: seq2048-b2-fits-with-bf16ce
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "n/a (consolidated into 71a45ae)"
verdict: crash
---

> *Backfilled from `RESULTS.tsv` + `OBSERVATIONS.md` approach-evolution.*

Tried seq=2048 batch=2 with bf16 CE (which saves ~2 GiB fp32 logits). **Crashed at compile time: `CompileTimeHbmOom` — 32.49 G vs 31.25 G limit, over by 1.25 GiB.**

## Hypothesis

bf16 CE saves enough HBM to promote seq=2048 from batch=1 (exp 9) to batch=2, doubling tokens/step.

## Result

Compile-time HBM exceeded by **1.25 GiB**. First data point on the **~1.25 GiB XLA compile-overhead pattern** that recurred across exp 11, 22, 23, and 32.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp10-splash-seq2048-b2-bf16ce](http://localhost:8791/?run=2026-04-23-gemma4-exp10-splash-seq2048-b2-bf16ce) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp10-splash-seq2048-b2-bf16ce`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp10-splash-seq2048-b2-bf16ce/`](../../../../../raw/profiles/2026-04-23-gemma4-exp10-splash-seq2048-b2-bf16ce/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: none (run did not reach training steps)
- **What's inside**: No runtime trace — compile-time HBM OOM (32.49 G vs 31.25 G limit). Directory holds the crash log and HLO dump fragment only.

## Verdict

**REJECTED — crash.** Not merged. The memory saving from bf16 CE wasn't enough to close the gap. Next attempt (exp 11) tried host-offload remat to break the compile-time budget; also failed.

## See also

- [exp 11 — offload remat attempt](2026-04-23-exp11-offload-remat-rejected.md) — the immediate follow-up.
- [exp 22 — batch=4 OOM](2026-04-23-exp22-batch4-fused-bwd-rejected.md), [exp 23 — seq=2048 b=2 OOM](2026-04-23-exp23-seq2048-batch2-fused-bwd-rejected.md) — same ceiling pattern.
- [exp 12 — seq=1024 b=2 bf16 CE](2026-04-23-exp12-bf16-ce-accepted.md) — the surviving bf16 CE story.

## Sources

- `RESULTS.tsv` row `exp10`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp10-splash-seq2048-b2-bf16ce/` — xprof run `2026-04-23-gemma4-exp10-splash-seq2048-b2-bf16ce` at http://localhost:8791/?run=2026-04-23-gemma4-exp10-splash-seq2048-b2-bf16ce

