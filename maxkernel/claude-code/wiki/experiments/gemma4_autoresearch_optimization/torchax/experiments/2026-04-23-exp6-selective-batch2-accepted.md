---
title: "Exp 6 — selective remat + batch=2 (KEEP, first TPS win +1.2 %)"
type: experiment
tags: [experiment, gemma4, batch, remat, selective, tps-win, hbm-ratchet]
hypothesis: batch2-on-selective-remat-memory-headroom
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "wiki:HEAD (exp5 code + --batch_size 2)"
verdict: supported
---

**First TPS win.** With exp 5's selective remat freeing 10+ GiB HBM, doubling batch from 1 to 2 yields 30,925 TPS (+1.2 % vs baseline 30,570) with loss trajectory preserved. Modest but real — unblocks the next round of memory-ratchet experiments.

## Hypothesis under test

**Statement**: Exp 5's selective remat freed ~10.4 GiB of HBM (95 % → 62 %). Doubling batch from 1 to 2 (HBM ratchet heuristic) fits, amortizes collective fixed costs across 2× tokens, and returns per-token cost below baseline.

## Setup

- Code unchanged from [exp 5](2026-04-23-exp5-selective-remat-accepted.md): `jax.checkpoint(forward_loss, policy=checkpoint_dots_with_no_batch_dims)`.
- Config change: `--batch_size 1` → `--batch_size 2`. Global batch 4 → 8 (fsdp=4, per-chip=2). Tokens/step 4096 → 8192.
- Command: `python -m train --steps 20 --batch_size 2 --seq_len 1024 ...`.

## Results

| Metric | Baseline | Exp 5 (sel-remat, b=1) | **Exp 6 (sel-remat, b=2)** | Δ vs baseline |
|---|---|---|---|---|
| Step time (wall) | 134.4 ms | 146.1 ms | **264.9 ms** | +97 % |
| Tokens/step | 4,096 | 4,096 | **8,192** | +2× |
| **TPS** | **30,570** | 28,035 | **30,925** | **+1.2 %** |
| Per-token cost (µs) | 32.8 | 35.7 | **32.3** | **−1.5 %** |
| Peak HBM | 29.69 GiB (95 %) | 19.32 GiB (62 %) | **25.92 GiB (83 %)** | −13 % |
| Stack reservation | 17.37 GiB | 7.01 GiB | 13.57 GiB | −22 % |
| Heap allocation | 12.31 GiB | 12.31 GiB | 12.35 GiB | flat |
| Free memory | 1.56 GiB | 11.93 GiB | 5.33 GiB | +3.8 GiB |
| Fragmentation | 0 % | 22 % | 48 % | |
| Loss trajectory | 3.93 → 1.97 | 3.93 → 1.94 | 3.82 → 1.57 (bigger-batch quality bonus) | parallel |

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp6-selective-batch2](http://localhost:8791/?run=2026-04-23-gemma4-exp6-selective-batch2) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp6-selective-batch2`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp6-selective-batch2/`](../../../../../raw/profiles/2026-04-23-gemma4-exp6-selective-batch2/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 5, 6, 7
- **What's inside**: xprof trace — selective remat + batch=2; first +TPS experiment.

## Mechanism

Step time went 146.1 → 264.9 ms for 2× tokens — ratio **1.81×**. Ideally the ratio would be exactly 2× (pure compute scaling) or less (amortization). The 1.81× means ~19 % of the batch=1 step was fixed overhead (collectives, compile artifacts, host-side dispatch) that amortizes; 81 % was compute that scales linearly.

Per-token cost: 146.1 / 4096 = 35.7 µs (batch=1) → 264.9 / 8192 = 32.3 µs (batch=2) = **−9.5 % per-token**. That is the amortization win, and it's enough to beat baseline's 32.8 µs/tok by 1.5 %.

Memory: stack reservation 7.0 → 13.6 GiB (+6.6 GiB added activation footprint for the doubled batch). Heap unchanged. Fragmentation went 22 % → 48 % — the batched activations refill some of the gaps, then leave new ones.

## Verdict

**SUPPORTED.** +1.2 % TPS over baseline with loss trajectory preserved. Modest win but real (σ on step time is 0.3 ms → < 0.1 % noise band — the 1.2 % is ~10× the noise). The exp 3 → exp 4 → exp 5 → exp 6 chain is validated.

## Next hypotheses

1. **Exp 7 — batch=3**: push further since 5.3 GiB is still free. *Result: discard — HBM 97 %, per-token efficiency degrades with memory pressure.*
2. **Exp 8 — splash attention via Pallas** (launched next): swap XLA SDPA for `jax.experimental.pallas.ops.tpu.splash_attention`. Targets `convolution fusion` 37 % and eliminates the N² score-matrix HBM traffic. Expected 15–40 % on attention. First pure-Pallas-kernel experiment.
3. **Revisit async-collective flags at batch=2**: exp 1 regressed at batch=1 because scheduler had no compute to hide collectives behind. At batch=2 there's 2× compute — flags may flip positive.
4. **Offload-variant remat** (`offload_dots_with_no_batch_dims`): move saved dots to host memory for even more HBM headroom; enables higher batch or splash-attention memory overhead.

## See also

- [program.md](../../program.md), [program page README](../../README.md).
- [OBSERVATIONS.md § exp06](OBSERVATIONS.md).
- [2026-04-23-exp5-selective-remat-accepted.md](2026-04-23-exp5-selective-remat-accepted.md) — the code change this experiment builds on.
- [2026-04-23-exp7-selective-batch3-rejected.md](2026-04-23-exp7-selective-batch3-rejected.md) — the follow-up that established batch=2 as the sweet spot.
- [rematerialization](../../../../concepts/rematerialization.md), [fsdp](../../../../concepts/fsdp.md).

## Sources

- `raw/profiles/2026-04-23-gemma4-exp6-selective-batch2/`
- `/tmp/gemma4_exp6.log`.
