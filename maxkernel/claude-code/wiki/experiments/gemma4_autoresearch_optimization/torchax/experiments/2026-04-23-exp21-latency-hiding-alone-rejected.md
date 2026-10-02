---
title: "Exp 21 — latency-hiding scheduler alone on splash+fused_bwd best (REJECTED, −16.6%)"
type: experiment
tags: [experiment, gemma4, xla-flags, scheduler, regression]
hypothesis: scheduler-alone-is-the-good-bit
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: b26e75c
verdict: refuted
---

> *Backfilled from `RESULTS.tsv` + commit `b26e75c`.*

Enabled `--xla_tpu_enable_latency_hiding_scheduler=true` on the splash+fused_bwd best-config (batch=2). **−16.6 % step time.** Same regression pattern as the full bundle in exp 1 / exp 13.

## Hypothesis

Maybe the exp 1 / exp 13 regression was driven by the scheduler alone, not the `async_collective_fusion*` flags. Isolating might reveal the innocent.

## Result

Refuted — flag alone still regresses at this workload + this config.

> **Later correction**: [exp 30](2026-04-23-exp30-latency-hiding-solo-potential.md) ran the same flag at the newer exp 25 stack (batch=3, SEQ_MINOR, block=1024) and found it **flat**. The exp 21 regression is therefore specific to the batch=2 stack snapshot — possibly interacting with some other diff in the config. The authoritative verdict on `latency_hiding_scheduler` is "flat in isolation" per exp 30. Keep exp 21 parked as a curious data point.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp21-latency-hiding](http://localhost:8791/?run=2026-04-23-gemma4-exp21-latency-hiding) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp21-latency-hiding`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp21-latency-hiding/`](../../../../../raw/profiles/2026-04-23-gemma4-exp21-latency-hiding/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — `--xla_tpu_enable_latency_hiding_scheduler=true` alone on splash+fused_bwd best (batch=2); regressed −16.6 % step time at this stack snapshot. Superseded by exp 30 at exp-25 stack where the same flag is flat.

## Verdict

**REJECTED** at the time; effectively **INCONCLUSIVE** in light of exp 30's later flat result. Not merged.

## See also

- [exp 30 — latency-hiding at exp 25 stack (flat)](2026-04-23-exp30-latency-hiding-solo-potential.md) — authoritative.
- [exp 1 — original bundle regression](2026-04-23-exp1-async-collective-flags-rejected.md).
- [exp 31 — overlap_compute_collective_tc alone (flat)](2026-04-23-exp31-overlap-compute-collective-tc-potential.md).

## Sources

- `RESULTS.tsv` row `exp21`.
- Commit `b26e75c`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp21-latency-hiding/` — xprof run `2026-04-23-gemma4-exp21-latency-hiding` at http://localhost:8791/?run=2026-04-23-gemma4-exp21-latency-hiding

