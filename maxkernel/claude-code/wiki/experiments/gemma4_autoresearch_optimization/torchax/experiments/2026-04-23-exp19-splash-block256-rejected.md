---
title: "Exp 19 — splash block sizes 512 → 256 (REJECTED, regression)"
type: experiment
tags: [experiment, gemma4, pallas, splash-attention, block-size, refuted]
hypothesis: smaller-block-reduces-vmem-pressure
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: ca80c11
verdict: refuted
---

> *Backfilled from `RESULTS.tsv` + commit `ca80c11` message.*

Tried `block_q = block_kv = 256` (halved from prior 512) to test whether smaller tiles would reduce VMEM pressure and increase occupancy. **Refuted — step time regressed (no number in the TSV; discarded on first run).**

## Hypothesis

Smaller tiles → less VMEM per tile → more tiles in flight → better pipelining. Expected 1–3 % win.

## Why it failed

Mirror of exp 29's later finding: splash's softmax accumulator is sequential across KV tiles. Smaller tiles mean more reload cycles with no concurrency benefit. MXU also prefers larger tiles (fewer dispatch overheads) up to VMEM limit.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp19-splash-block256](http://localhost:8791/?run=2026-04-23-gemma4-exp19-splash-block256) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp19-splash-block256`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp19-splash-block256/`](../../../../../raw/profiles/2026-04-23-gemma4-exp19-splash-block256/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — splash `block_q=block_kv=256`; smaller tiles regressed (softmax accumulator is sequential across KV tiles).

## Verdict

**REJECTED.** Not merged. The correct direction turned out to be **larger** blocks (exp 25: `block=1024`), not smaller.

## See also

- [exp 25 — splash block=1024 (current best)](2026-04-23-exp25-splash-block1024-accepted.md) — the winning direction.
- [exp 29 — asymmetric blocks (also refuted)](2026-04-23-exp29-splash-asymmetric-rejected.md) — later, explicit mechanism analysis.

## Sources

- `RESULTS.tsv` row `exp19`.
- Commit `ca80c11`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp19-splash-block256/` — xprof run `2026-04-23-gemma4-exp19-splash-block256` at http://localhost:8791/?run=2026-04-23-gemma4-exp19-splash-block256

