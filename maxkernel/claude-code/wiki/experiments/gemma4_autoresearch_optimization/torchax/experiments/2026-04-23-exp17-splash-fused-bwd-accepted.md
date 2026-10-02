---
title: "Exp 17 — splash use_fused_bwd_kernel=True (ACCEPTED, +0.9%)"
type: experiment
tags: [experiment, gemma4, pallas, splash-attention, fused-bwd, tps-win]
hypothesis: fused-bwd-saves-dispatch-overhead
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: 6e0f354
verdict: supported
---

> *Backfilled from `RESULTS.tsv` + commit `6e0f354` + merge commit `e71cdfc`.*

Correct `use_fused_bwd_kernel=True` config — dq block-size parameters **omitted** (they're forbidden with fused bwd; [exp 16](2026-04-23-exp16-fused-bwd-broken-rejected.md) hit the error and fell back to XLA). **Result: 32,663 TPS, +0.9 % over exp 12's 32,340. New best.**

## Hypothesis

Splash's fused backward kernel combines dQ/dK/dV into one kernel call, saving ~2.5 ms/step of per-op dispatch + reload overhead.

## Setup

```diff
 block_sizes = sa_kernel.BlockSizes(
     block_q=block_q,
     block_kv=block_kv,
     block_kv_compute=block_kv_compute,
     block_q_dkv=min(1024, seq_len),
     block_kv_dkv=min(1024, seq_len),
     block_kv_dkv_compute=min(1024, seq_len),
-    block_q_dq=<prior value>,
-    block_kv_dq=<prior value>,
+    # block_q_dq / block_kv_dq intentionally omitted
+    use_fused_bwd_kernel=True,
 )
```

## Results

| Metric | Exp 12 | **Exp 17** | Δ |
|---|---|---|---|
| TPS | 32,340 | **32,663** | **+0.9 %** (+6.8 % vs baseline) |
| Step time | 253.3 ms | 250.8 ms | −1.0 % |
| Loss | clean | match | identical |

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp17-splash-fused-bwd](http://localhost:8791/?run=2026-04-23-gemma4-exp17-splash-fused-bwd) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp17-splash-fused-bwd`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp17-splash-fused-bwd/`](../../../../../raw/profiles/2026-04-23-gemma4-exp17-splash-fused-bwd/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — splash `use_fused_bwd_kernel=True` with correct block-size config; ~2.5 ms/step saved on bwd pass.

## Verdict

**SUPPORTED.** Merged on branch `perfautoresearch/v6e4-20260423-exp17-splash-fused-bwd`, then merged to trunk (commit `e71cdfc`).

## See also

- [exp 16 — fused_bwd with leftover dq params (invalid)](2026-04-23-exp16-fused-bwd-broken-rejected.md) — the cautionary predecessor.
- [exp 18 — fused_bwd + batch=3](2026-04-23-exp18-fused-bwd-batch3-accepted.md) — the next ratchet.

## Sources

- `RESULTS.tsv` row `exp17`.
- Commits `6e0f354`, `e71cdfc`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp17-splash-fused-bwd/` — xprof run `2026-04-23-gemma4-exp17-splash-fused-bwd` at http://localhost:8791/?run=2026-04-23-gemma4-exp17-splash-fused-bwd

