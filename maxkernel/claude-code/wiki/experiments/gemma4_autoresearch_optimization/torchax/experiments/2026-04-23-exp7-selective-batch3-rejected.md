---
title: "Exp 7 — selective remat + batch=3 (DISCARD, HBM-pressure degrades per-token)"
type: experiment
tags: [experiment, gemma4, batch, remat, selective, discard, hbm-ratchet]
hypothesis: push-batch3-with-selective-remat
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "wiki:HEAD (exp5 code + --batch_size 3)"
verdict: refuted
---

Attempted to push batch from 2 to 3 on top of exp 5/6's selective remat. Per-token efficiency **degraded** (32.3 → 33.7 µs) instead of improving — memory hit 97.6 % (0.74 GiB free), the allocator paid for it. **batch=2 is the sweet spot** at this config.

## Hypothesis under test

**Statement**: Exp 6 (batch=2) left 5.3 GiB free (83 % HBM). Batch=3 should fit (adds ~5 GiB activations → ~31 GiB, right at ceiling) and further amortize the collective fixed costs. Per-token cost target < 32.3 µs.

## Setup

- Code unchanged from exp 5 / 6.
- Config: `--batch_size 3 --seq_len 1024`. Global batch 12 (per-chip 3). Tokens/step 12,288.
- Command: `python -m train --steps 20 --batch_size 3 --seq_len 1024 ...`.

## Results

| Metric | Baseline | Exp 6 (b=2) | **Exp 7 (b=3)** | Δ vs exp 6 | Δ vs baseline |
|---|---|---|---|---|---|
| Step time (wall) | 134.4 ms | 264.9 ms | 413.5 ms | +56 % | +208 % |
| Tokens/step | 4,096 | 8,192 | **12,288** | +1.5× | +3× |
| TPS | 30,570 | 30,925 | **29,720** | **−3.9 %** | **−2.8 %** |
| Per-token (µs) | 32.8 | 32.3 | **33.7** | **+4.3 %** | +2.7 % |
| Peak HBM | 29.69 GiB (95 %) | 25.92 GiB (83 %) | **30.50 GiB (97.6 %)** | +18 % | +3 % |
| Stack reservation | 17.37 | 13.57 | 18.21 | | |
| Heap allocation | 12.31 | 12.35 | 12.29 | flat | flat |
| Free memory | 1.56 GiB | 5.33 GiB | **0.74 GiB** | | |
| Fragmentation | 0 % | 48 % | **0 %** | | |

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp7-selective-batch3](http://localhost:8791/?run=2026-04-23-gemma4-exp7-selective-batch3) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp7-selective-batch3`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp7-selective-batch3/`](../../../../../raw/profiles/2026-04-23-gemma4-exp7-selective-batch3/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 5, 6, 7
- **What's inside**: xprof trace — selective remat + batch=3; HBM 97.6 % at ceiling, degraded per-token cost.

## Mechanism

Step-time scaling batch=2 → batch=3: 264.9 → 413.5 ms (ratio **1.56×** for 1.5× tokens). Near-linear, slightly worse than projection. If amortization were the dominant effect we'd expect < 1.5× ratio; actual is slightly > 1.5×.

Per-token cost went **up** by 4.3 %. Two concurrent causes:
1. **HBM near ceiling (97.6 %)**: the allocator has almost no slack; each kernel launch pays for small reallocations / compile-time scheduling around tight budgets. Fragmentation jumped back to 0 % (batch=3 packs the heap), meaning the allocator had to work harder to place every tensor.
2. **Linear compute scaling exceeds amortization gain**: at batch=2 the amortization of fixed overhead (~19 % per exp 6 analysis) was the win; at batch=3 the additional amortization would save ~5–7 %, but memory-pressure cost ate that and more.

The pattern: **there's a batch sweet spot set by HBM pressure, not by pure amortization math**. At this config, batch=2 is optimal. To push past, need another memory win first.

## Verdict

**REFUTED.** Revert config to batch=2.

## Next hypotheses

1. **Splash attention via Pallas** (exp 8, launched): targets `convolution fusion` 37 % and the N² score-matrix HBM traffic. Frees memory AND speed; may enable batch>2 as a side effect.
2. **`offload_dots_with_no_batch_dims`**: variant of selective remat that offloads saved dots to host memory. Frees HBM at cost of PCIe transfer. Would make batch=4 feasible.
3. **Drop fp32 CE upcast**: the hand-rolled CE in `forward_loss` upcasts logits to fp32 (saves ~4 GiB fp32 `[B, S, V=262144]` tensor at batch=2). Simple memory win (not Pallas but profile-driven).
4. **Async-collective flags at batch=2** (revisit exp 1): scheduler has 2× compute to hide collectives behind now; may flip positive.

## See also

- [program.md](../../program.md), [program page README](../../README.md).
- [OBSERVATIONS.md § exp07](OBSERVATIONS.md).
- [2026-04-23-exp6-selective-batch2-accepted.md](2026-04-23-exp6-selective-batch2-accepted.md) — the current-best config this experiment reverts to.
- [rematerialization concept](../../../../concepts/rematerialization.md).

## Sources

- `raw/profiles/2026-04-23-gemma4-exp7-selective-batch3/`
- `/tmp/gemma4_exp7.log`.
