---
title: "Exp 14 — splash + bf16 CE at seq=2048 b=1 (ACCEPTED, saves 2.5% vs fp32 CE)"
type: experiment
tags: [experiment, gemma4, bf16, cross-entropy, long-seq]
hypothesis: bf16-ce-saves-at-seq2048-too
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "n/a (consolidated into 71a45ae)"
verdict: supported
---

> *Backfilled from `RESULTS.tsv` row.*

Cross-point experiment: take [exp 12](2026-04-23-exp12-bf16-ce-accepted.md)'s splash+bf16 CE stack and apply it to seq=2048 b=1 instead of seq=1024 b=2. Result: **31,960 TPS at 256.3 ms/step** — saves ~2.5 % step time vs [exp 9](2026-04-23-exp9-splash-seq2048-accepted.md)'s fp32 CE version (263 ms).

## Hypothesis

bf16 CE should work the same at seq=2048 as at seq=1024. Matrix-fill data point.

## Result

Confirmed. TPS 31,960 — just below exp 12's 32,340 on the same total tokens/step (8192 for both: 2×1024 at b=2 and 1×2048 at b=1). seq=1024 b=2 edges seq=2048 b=1 by ~1.2 %.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp14-splash-bf16ce-seq2048](http://localhost:8791/?run=2026-04-23-gemma4-exp14-splash-bf16ce-seq2048) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp14-splash-bf16ce-seq2048`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp14-splash-bf16ce-seq2048/`](../../../../../raw/profiles/2026-04-23-gemma4-exp14-splash-bf16ce-seq2048/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — splash + bf16 CE at seq=2048 b=1; saves ~2.5 % step time vs exp 9's fp32 CE.

## Verdict

**SUPPORTED.** Merged to trunk. Long-seq users get bf16 CE for free.

## See also

- [exp 9 — splash at seq=2048 (fp32 CE)](2026-04-23-exp9-splash-seq2048-accepted.md) — direct predecessor.
- [exp 12 — bf16 CE at seq=1024 b=2](2026-04-23-exp12-bf16-ce-accepted.md) — sibling comparison.
- [exp 28 — full exp25 stack at seq=2048 b=1](2026-04-23-exp28-seq2048-exp25config-accepted.md) — successor.

## Sources

- `RESULTS.tsv` row `exp14`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp14-splash-bf16ce-seq2048/` — xprof run `2026-04-23-gemma4-exp14-splash-bf16ce-seq2048` at http://localhost:8791/?run=2026-04-23-gemma4-exp14-splash-bf16ce-seq2048

