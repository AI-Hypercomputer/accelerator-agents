---
title: "Exp 9 — splash attention at seq=2048 batch=1 (ACCEPTED, +1.9% + fixed NaN)"
type: experiment
tags: [experiment, gemma4, pallas, splash-attention, long-seq, numerical-stability]
hypothesis: splash-at-seq2048
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "n/a (pre-branching-model, consolidated into 71a45ae)"
verdict: supported
---

> *Backfilled from `RESULTS.tsv` + `OBSERVATIONS.md` threading + commit `71a45ae` narrative. The original run predates the per-experiment branching discipline; the granular pass happens to be captured in the aggregate-commit but not in a dedicated writeup until now.*

Ran [exp 8](2026-04-23-exp8-splash-attention-accepted.md)'s splash kernel stack at **seq=2048, batch=1** — the configuration that had crashed with NaN loss since the baseline. **Result: 31,148 TPS (+1.9 % over baseline-seq1024), loss 3.26 → 1.50 clean descent.** Splash's online-max softmax is numerically stable at seq=2048 — **NaN bug fixed**. Unblocks long-seq training.

## Hypothesis under test

**Statement**: Splash's online-max softmax is numerically stable by construction (running max subtracted before exp). The NaN-at-seq≥2048 bug observed with XLA SDPA should not reproduce with splash.

## Setup

- Config: same as exp 8 (selective remat + splash_pallas) but `--seq_len 2048 --batch_size 1`.
- Profile: `raw/profiles/2026-04-23-gemma4-exp9-splash-seq2048/` (gitignored).

## Results

| Metric | Baseline-seq2048 (NaN) | **Exp 9 (splash seq=2048 b=1)** |
|---|---|---|
| TPS | 32,900 (but NaN loss — invalid) | **31,148** |
| Step time | 249 ms | 263 ms |
| Peak HBM | 29.69 GiB | 25.76 GiB |
| Loss trajectory | NaN | **3.26 → 1.50** (clean) |

+1.9 % TPS vs baseline-seq1024 (30,570). Strict upgrade over the NaN-broken baseline-seq2048.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp9-splash-seq2048](http://localhost:8791/?run=2026-04-23-gemma4-exp9-splash-seq2048) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp9-splash-seq2048`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp9-splash-seq2048/`](../../../../../raw/profiles/2026-04-23-gemma4-exp9-splash-seq2048/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — splash at seq=2048 b=1; NaN bug fixed, clean loss descent.

## Verdict

**SUPPORTED.** Splash fixes the NaN and gives real throughput at seq=2048. Merged to trunk.

## See also

- [exp 8 — splash attention](2026-04-23-exp8-splash-attention-accepted.md) — the kernel introduction.
- [exp 14 — splash + bf16 CE at seq=2048 b=1](2026-04-23-exp14-splash-seq2048-bf16ce-accepted.md).
- [exp 28 — seq=2048 b=1 at full exp25 stack](2026-04-23-exp28-seq2048-exp25config-accepted.md).

## Sources

- `RESULTS.tsv` row `exp09`.
- `raw/profiles/2026-04-23-gemma4-exp9-splash-seq2048/`.
- `OBSERVATIONS.md` approach-evolution section.
