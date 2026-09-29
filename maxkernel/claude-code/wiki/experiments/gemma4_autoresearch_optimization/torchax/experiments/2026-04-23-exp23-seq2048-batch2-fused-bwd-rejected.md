---
title: "Exp 23 — seq=2048 b=2 with fused_bwd (REJECTED — same 1.25 GiB ceiling)"
type: experiment
tags: [experiment, gemma4, oom, long-seq, batch-growth, hbm-ceiling]
hypothesis: fused-bwd-unblocks-seq2048-b2
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: 4164f12
verdict: crash
---

> *Backfilled from `RESULTS.tsv` + commit `4164f12` message.*

Tried seq=2048 batch=2 with fused_bwd. **OOM at compile time — same 1.25 GiB threshold as exp 10, exp 11, exp 22.**

## Hypothesis

fused_bwd reduces backward-pass memory; maybe enough to unblock seq=2048 b=2 (the spot that's OOM'd with every other config).

## Result

No. The 1.25 GiB gap is structural.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp23-seq2048-b2-fused-bwd](http://localhost:8791/?run=2026-04-23-gemma4-exp23-seq2048-b2-fused-bwd) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp23-seq2048-b2-fused-bwd`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp23-seq2048-b2-fused-bwd/`](../../../../../raw/profiles/2026-04-23-gemma4-exp23-seq2048-b2-fused-bwd/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: none (run did not reach training steps)
- **What's inside**: No runtime trace — compile-time HBM OOM at same 1.25 GiB threshold.

## Verdict

**REJECTED — crash.** Not merged. The same lesson as exp 22: batch/seq ratchet is blocked by XLA compile-time accounting; changing which kernel handles the bwd doesn't help.

## See also

- [exp 10](2026-04-23-exp10-seq2048-batch2-bf16ce-rejected.md), [exp 11](2026-04-23-exp11-offload-remat-rejected.md), [exp 22](2026-04-23-exp22-batch4-fused-bwd-rejected.md) — same 1.25 GiB ceiling.

## Sources

- `RESULTS.tsv` row `exp23`.
- Commit `4164f12`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp23-seq2048-b2-fused-bwd/` — xprof run `2026-04-23-gemma4-exp23-seq2048-b2-fused-bwd` at http://localhost:8791/?run=2026-04-23-gemma4-exp23-seq2048-b2-fused-bwd

