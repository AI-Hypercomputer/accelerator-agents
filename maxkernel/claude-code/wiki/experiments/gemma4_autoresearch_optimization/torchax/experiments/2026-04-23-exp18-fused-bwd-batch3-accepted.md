---
title: "Exp 18 — fused_bwd + batch=3 (ACCEPTED, +8.0% new best)"
type: experiment
tags: [experiment, gemma4, pallas, splash-attention, fused-bwd, batch-growth, tps-win]
hypothesis: fused-bwd-stacks-with-batch3
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: e4d16dc
verdict: supported
---

> *Backfilled from `RESULTS.tsv` + merge commit `e4d16dc`.*

Stack [exp 17](2026-04-23-exp17-splash-fused-bwd-accepted.md) (fused_bwd) with [exp 15](2026-04-23-exp15-splash-bf16ce-batch3-accepted.md)'s batch=3. **Result: 33,016 TPS, +8.0 % over baseline — new best.**

## Hypothesis

fused_bwd's ~2.5 ms/step savings + batch=3's amortization multiply. No surprise — orthogonal optimizations compose.

## Results

| Metric | Exp 15 (batch=3 without fused_bwd) | **Exp 18 (batch=3 + fused_bwd)** | Δ |
|---|---|---|---|
| TPS | 32,717 | **33,016** | **+0.9 %** (+8.0 % vs baseline) |
| Step time | 375.5 ms | ~372 ms | −0.9 % |
| HBM | 30.87 GiB | similar (still near ceiling) | ~flat |

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp18-fused-bwd-batch3](http://localhost:8791/?run=2026-04-23-gemma4-exp18-fused-bwd-batch3) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp18-fused-bwd-batch3`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp18-fused-bwd-batch3/`](../../../../../raw/profiles/2026-04-23-gemma4-exp18-fused-bwd-batch3/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — fused_bwd + batch=3 on splash+bf16 CE; +8.0 % over baseline new best at the time.

## Verdict

**SUPPORTED.** Merged to trunk (commit `e4d16dc`). Current best at this point.

## See also

- [exp 15 — batch=3 without fused_bwd](2026-04-23-exp15-splash-bf16ce-batch3-accepted.md).
- [exp 17 — fused_bwd at batch=2](2026-04-23-exp17-splash-fused-bwd-accepted.md).
- [exp 24 — SEQ_MINOR layout](2026-04-23-exp24-splash-seq-minor-accepted.md) — the next ratchet.

## Sources

- `RESULTS.tsv` row `exp18`.
- Commit `e4d16dc`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp18-fused-bwd-batch3/` — xprof run `2026-04-23-gemma4-exp18-fused-bwd-batch3` at http://localhost:8791/?run=2026-04-23-gemma4-exp18-fused-bwd-batch3

