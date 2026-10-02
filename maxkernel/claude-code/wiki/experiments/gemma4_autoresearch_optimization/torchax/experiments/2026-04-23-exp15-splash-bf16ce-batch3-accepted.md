---
title: "Exp 15 — splash + bf16 CE at batch=3 (ACCEPTED, +7.0% new best)"
type: experiment
tags: [experiment, gemma4, batch-growth, memory-ceiling, tps-win]
hypothesis: bf16-ce-frees-enough-hbm-for-batch3
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "n/a (consolidated into 71a45ae)"
verdict: supported
---

> *Backfilled from `RESULTS.tsv` row.*

bf16 CE saves ~1.5 GiB of fp32-logits materialization. [Exp 7](2026-04-23-exp7-selective-batch3-rejected.md) had tried batch=3 at 97.6 % HBM and degraded per-token efficiency. With bf16 CE's headroom, **batch=3 now fits — 32,717 TPS, 375.5 ms/step, HBM 98.78 % right at the ceiling. +7.0 % over baseline, new best.**

## Hypothesis

The HBM-ratchet heuristic: every memory saving (selective remat, bf16 CE) unlocks the next batch size. batch=3 was previously blocked by HBM; bf16 CE might close the gap.

## Results

| Metric | Exp 12 (batch=2) | **Exp 15 (batch=3)** | Δ |
|---|---|---|---|
| TPS | 32,340 | **32,717** | **+1.2 %** (+7.0 % vs baseline) |
| Step time | 253.3 ms | 375.5 ms | +48 % (but more tokens/step) |
| Peak HBM | 24.79 GiB (79 %) | 30.87 GiB (98.78 %) | at ceiling |
| Per-token cost | 30.92 µs | 30.56 µs | −1.2 % |
| Loss descent | healthy | healthy | match |

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp15-splash-b3-bf16ce](http://localhost:8791/?run=2026-04-23-gemma4-exp15-splash-b3-bf16ce) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp15-splash-b3-bf16ce`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp15-splash-b3-bf16ce/`](../../../../../raw/profiles/2026-04-23-gemma4-exp15-splash-b3-bf16ce/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — splash + bf16 CE + batch=3; HBM 98.78 % at ceiling but per-token cost improved.

## Verdict

**SUPPORTED.** Merged. Current best at this point. Per-token cost improved further with fused_bwd in exp 18.

## See also

- [exp 7 — batch=3 without bf16 CE](2026-04-23-exp7-selective-batch3-rejected.md) — the predecessor that OOM'd before bf16 CE.
- [exp 12 — bf16 CE at batch=2](2026-04-23-exp12-bf16-ce-accepted.md) — the memory freeing.
- [exp 18 — fused_bwd + batch=3](2026-04-23-exp18-fused-bwd-batch3-accepted.md) — the next ratchet.

## Sources

- `RESULTS.tsv` row `exp15`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp15-splash-b3-bf16ce/` — xprof run `2026-04-23-gemma4-exp15-splash-b3-bf16ce` at http://localhost:8791/?run=2026-04-23-gemma4-exp15-splash-b3-bf16ce

