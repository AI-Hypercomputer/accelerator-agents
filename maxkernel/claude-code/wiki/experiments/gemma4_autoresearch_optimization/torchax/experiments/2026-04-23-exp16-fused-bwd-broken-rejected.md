---
title: "Exp 16 — use_fused_bwd_kernel=True with stale dq block params (REJECTED — splash errored, XLA fallback)"
type: experiment
tags: [experiment, gemma4, pallas, splash-attention, fused-bwd, invalid, xla-fallback]
hypothesis: use-fused-bwd-kernel-works-as-is
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: bcaef89
verdict: invalid
---

> *Backfilled from `RESULTS.tsv` row + commit `bcaef89` message.*

Set `use_fused_bwd_kernel=True` on the splash kernel but left the `block_q_dq` / `block_kv_dq` parameters configured. **Splash errored: `ValueError: Block sizes for dq kernel are not needed with a fused kernel`. Our XLA fallback path fired for every layer — the measured 32,027 TPS / 255.8 ms is XLA-attention, not splash-fused-bwd.**

## Hypothesis

The `use_fused_bwd_kernel=True` flag should combine Q/K/V backward into a single kernel call, saving dispatch overhead.

## What went wrong

The dq block-size parameters must be **omitted** (not set) when `use_fused_bwd_kernel=True`. My edit left them in place; splash raised at build time; my try/except in `splash_attention_fn` caught the error and fell through to the plain XLA attention path — invisibly, per call.

## Interesting side-finding

The XLA fallback number (32,027 TPS, 255.8 ms) is **within 2 ms** of the real splash path at this config. XLA's attention at seq=1024 is competitive with splash — the splash win scales with seq² and kicks in at longer sequences.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp16-splash-fused-bwd](http://localhost:8791/?run=2026-04-23-gemma4-exp16-splash-fused-bwd) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp16-splash-fused-bwd`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp16-splash-fused-bwd/`](../../../../../raw/profiles/2026-04-23-gemma4-exp16-splash-fused-bwd/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — `use_fused_bwd_kernel=True` with stale dq block params; splash errored + XLA fallback fired for every attention layer. This profile measures XLA attention, not splash-fused-bwd — interesting side-finding.

## Verdict

**REJECTED / INVALID.** This run measured the wrong kernel. The correct config landed in [exp 17](2026-04-23-exp17-splash-fused-bwd-accepted.md) with the dq params stripped. Not merged.

## See also

- [exp 17 — fused_bwd done right](2026-04-23-exp17-splash-fused-bwd-accepted.md).
- [exp 8 — original splash integration](2026-04-23-exp8-splash-attention-accepted.md).

## Sources

- `RESULTS.tsv` row `exp16`.
- Commit `bcaef89`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp16-splash-fused-bwd/` — xprof run `2026-04-23-gemma4-exp16-splash-fused-bwd` at http://localhost:8791/?run=2026-04-23-gemma4-exp16-splash-fused-bwd

