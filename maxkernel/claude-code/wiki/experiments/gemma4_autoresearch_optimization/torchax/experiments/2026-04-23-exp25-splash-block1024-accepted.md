---
title: "Exp 25 — splash block sizes 512 → 1024 (ACCEPTED, +0.6% NEW BEST +9.2% over baseline)"
type: experiment
tags: [experiment, gemma4, pallas, splash-attention, block-size, tps-win, session-best]
hypothesis: whole-seq-one-block-attention
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: bc26a7b
verdict: supported
---

> *Backfilled from `RESULTS.tsv` + commits `bc26a7b`, `ebb00ec`. This is the **session-best** configuration — everything on trunk after this merge reproduces this number.*

Bumped splash block sizes from 512 to 1024 at seq=1024 (whole-sequence one-block attention). **Result: 33,372 TPS, +0.6 % over exp 24, +9.2 % over baseline-seq1024. NEW BEST.**

## Hypothesis

At seq=1024, `block_q = block_kv = 1024` covers the entire sequence in a single tile — maximizes reuse of Q/K/V loads, minimizes dispatch overhead. Earlier exp 19 tried the opposite direction (block=256) and regressed; larger blocks are favored at this MXU-ridge regime.

## Setup

```diff
-    block_q = min(512, seq_len)
-    block_kv = min(512, seq_len)
-    block_kv_compute = min(512, seq_len)
+    block_q = min(1024, seq_len)
+    block_kv = min(1024, seq_len)
+    block_kv_compute = min(1024, seq_len)
```

Also matching `block_q_dkv / block_kv_dkv / block_kv_dkv_compute` bumped to 1024.

## Results

| Metric | Exp 24 (block=512, SEQ_MINOR) | **Exp 25 (block=1024, SEQ_MINOR)** | Δ |
|---|---|---|---|
| TPS | 33,193 | **33,372** | **+0.5 %** (+9.2 % vs baseline) |
| Step time | ~370 ms | ~368 ms | −0.5 % |
| Loss | clean | match | identical |

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp25-splash-block1024](http://localhost:8791/?run=2026-04-23-gemma4-exp25-splash-block1024) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp25-splash-block1024`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp25-splash-block1024/`](../../../../../raw/profiles/2026-04-23-gemma4-exp25-splash-block1024/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — **session-best config**: splash block=1024 + SEQ_MINOR + fused_bwd + bf16 CE + selective remat + batch=3. 33,372 TPS, +9.2 % over baseline.

## Verdict

**SUPPORTED.** Merged to trunk (commit `ebb00ec`).

## Current best config (trunk after this commit)

- Hardware: v6e-4
- Mesh: 1D fsdp=4
- Precision: bf16 (weights, activations, CE, log-softmax)
- Config: `--seq_len 1024 --batch_size 3 --strategy fsdp`
- Remat: `jax.checkpoint(forward_loss, policy=checkpoint_dots_with_no_batch_dims)`
- Attention: splash_pallas via `torchax.interop.call_jax` + `jax.shard_map`, `block_q = block_kv = 1024`, `use_fused_bwd_kernel=True`, `QKVLayout.SEQ_MINOR`
- CE: bf16 log-softmax (safe due to `final_logit_softcapping=30.0`)

## Subsequent experiments (exp 26–33) failed to improve further

See [the ceiling analysis](../../../../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md) for the full trajectory. Structural attempts (scan-over-layers, 2D mesh, Pallas RMSNorm) either blocked or regressed.

## See also

- [exp 24 — SEQ_MINOR (predecessor)](2026-04-23-exp24-splash-seq-minor-accepted.md).
- [exp 19 — block=256 (refuted opposite direction)](2026-04-23-exp19-splash-block256-rejected.md).
- [exp 29 — asymmetric blocks 1024/512 (refuted)](2026-04-23-exp29-splash-asymmetric-rejected.md) — post-exp 25 sanity check.
- [Session ceiling analysis](../../../../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md) — full-trajectory synthesis.

## Sources

- `RESULTS.tsv` row `exp25`.
- Commits `bc26a7b` (direct), `ebb00ec` (merge to trunk).
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp25-splash-block1024/` — xprof run `2026-04-23-gemma4-exp25-splash-block1024` at http://localhost:8791/?run=2026-04-23-gemma4-exp25-splash-block1024

