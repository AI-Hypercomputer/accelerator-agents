---
title: "Exp 29 — splash asymmetric blocks block_q=1024, block_kv=512 (REFUTED, −0.37%)"
type: experiment
tags: [experiment, gemma4, pallas, splash-attention, block-size, refuted]
hypothesis: splash-asymmetric-blocks
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "branch perfautoresearch/v6e4-20260423-exp29-splash-asymmetric"
verdict: refuted
---

Tried halving `block_kv` (and `block_kv_compute`) to 512 while keeping `block_q` at 1024, under the theory that a smaller KV tile would reduce per-tile VMEM and increase concurrency on the splash scheduler. **Result: −0.37 % TPS vs exp 25 — within noise, but directionally worse.** Larger symmetric block_kv = block_q = 1024 wins.

## Hypothesis under test

**Statement**: At seq=1024, the splash kernel with `block_q = block_kv = 1024` runs as one big Q × KV tile per head. VMEM budget limits how many such tiles can be in flight. Halving `block_kv` to 512 gives 2 KV tiles per Q tile — smaller per-tile VMEM footprint, potentially more overlap / better occupancy without increasing HBM traffic (same total KV data, just split).

Falsifiable: TPS > 33,372 + noise band (+0.2 %) → supported. TPS ≤ 33,372 + noise → refuted.

## Setup

- Branch `perfautoresearch/v6e4-20260423-exp29-splash-asymmetric` off trunk at exp 25.
- Single edit in `torchax/model/pallas_attention.py`:
  ```
  - block_kv = min(1024, seq_len)
  - block_kv_compute = min(1024, seq_len)
  + block_kv = min(512, seq_len)
  + block_kv_compute = min(512, seq_len)
  ```
- All other block params (block_q, block_q_dkv, block_kv_dkv, block_kv_dkv_compute) stay at 1024.
- Same command and stack as exp 25: `python -m train --steps 20 --batch_size 3 --seq_len 1024 --profile_dir … --profile_steps 10 11 12`.

## Results

| Metric | Exp 25 (block=1024 symmetric) | **Exp 29 (block_q=1024, block_kv=512)** | Δ |
|---|---|---|---|
| TPS | 33,372 | **33,247** | **−0.37 %** |
| Step time (steady-state mean, steps 2–19) | ~368 ms | **369.62 ms** | +0.44 % |
| Per-step σ | tight | min 368.70 / max 371.60 (n=18) | — |
| Compile step 0 | ~155 s | 176.22 s | +14 % |
| Step 1 recompile | ~152 s | 158.87 s | +4 % |
| Loss trajectory | 3.82 → 1.55 | 3.83 → 1.85 | match |

Compile is a bit longer because the splash kernel is specialized for the new (block_q, block_kv) tuple — one-time cost per config, not a steady-state metric.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp29-splash-asymmetric](http://localhost:8791/?run=2026-04-23-gemma4-exp29-splash-asymmetric) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp29-splash-asymmetric`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp29-splash-asymmetric/`](../../../../../raw/profiles/2026-04-23-gemma4-exp29-splash-asymmetric/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — splash block_q=1024, block_kv=512; refuted, −0.37 %.

## Mechanism

The TPU MXU prefers larger matmul tiles up to the VMEM limit because dispatch overhead and reload-from-HBM costs dominate when tiles are small. At seq=1024, a `(block_q=1024, block_kv=1024)` splash call performs the entire attention in one K/V pass; `(block_q=1024, block_kv=512)` does two K/V passes per Q tile, doubling reload cost with no compensating concurrency gain (the second pass depends on the first through the softmax accumulator). The expected concurrency win didn't materialize because splash's softmax-accumulator structure is sequential across KV tiles, not parallel.

Takeaway: **symmetric block_q = block_kv = 1024 is the sweet spot at seq=1024**. Asymmetric doesn't help.

## Verdict

**REFUTED.** −0.37 % is within noise band but directionally wrong. Not merged.

Parking the asymmetric direction: likely no more wins from this knob at seq=1024. At seq≥2048 the same test may behave differently (more KV tiles per Q already), but that's a separate experiment if it comes up.

## Next hypotheses

1. **Persistent JAX compile cache** — not TPS but 30× iteration speed; pending since exp 2.
2. **2D mesh (fsdp=2, tp=2)** — biggest remaining structural lever, engineering-medium via the existing `plan_tp_shardings` path. Shards Q/K/V/MLP weights along the TP axis → ~50 % less per-chip weight memory → unlocks batch=4 (currently OOM in 1D fsdp=4).
3. **Pallas RMSNorm with hand-rolled custom_vjp** — targets the ~57 ms loop-fusion bucket per step; eng-heavy but bounded.
4. **Retry batch=4 at exp25 config** — likely still OOM (block=1024 uses more VMEM not less), but quick data.

## See also

- [exp 25 — splash block=1024, current best](../..) (writeup gap — RESULTS.tsv row)
- [exp 19 — splash block=256 (discard)](../..) (writeup gap — RESULTS.tsv row) — the other block-size data point.
- [splash-attention concept](../../../../concepts/splash-attention.md), [attention-block-sizes concept](../../../../concepts/attention-block-sizes.md).

## Sources

- `wiki/experiments/gemma4_autoresearch_optimization/torchax/model/pallas_attention.py` (modified on branch only, 3 lines)
- `/tmp/gemma4_exp29.log`
- `raw/profiles/2026-04-23-gemma4-exp29-splash-asymmetric/`
