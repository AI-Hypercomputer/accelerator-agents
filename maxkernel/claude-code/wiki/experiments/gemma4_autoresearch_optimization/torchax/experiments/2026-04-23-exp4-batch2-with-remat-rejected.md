---
title: "Exp 4 — double batch with full remat (DISCARD, chain net −9 % TPS)"
type: experiment
tags: [experiment, gemma4, batch, remat, discard, hbm-ratchet]
hypothesis: double-batch-unlocked-by-remat
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "wiki:HEAD (exp3 code + --batch_size 2)"
verdict: refuted
---

Doubled batch from 1 to 2 (on top of exp 3's full-remat code change), expecting the per-step amortization of 2× tokens to outweigh remat's compute tax. Didn't: per-token cost is still above baseline. Chain exp 3 + exp 4 is net **−9 % TPS** vs pre-remat baseline.

## Hypothesis under test

**Statement**: Exp 3 freed ~8.6 GiB of HBM (95 % → 67 %). With that headroom, doubling batch from 1 to 2 at seq=1024 fits, amortizes the per-step collective overhead across 2× tokens, and produces a TPS win over the pre-remat baseline.

Origin: `program.md` HBM-ratchet heuristic — "when HBM drops, immediately try the blocked state-growing change."

## Setup

- Code unchanged from exp 3 (`grad_fn = jax.value_and_grad(jax.checkpoint(forward_loss))`).
- Config change: `--batch_size 1` → `--batch_size 2`. Global batch 4 → 8, tokens/step 4,096 → 8,192.
- Command: `python -m train --steps 20 --batch_size 2 --seq_len 1024 ...`.

## Results

| Metric | Baseline | Exp 3 (batch=1 + remat) | Exp 4 (batch=2 + remat) | Δ vs baseline | Δ vs exp 3 |
|---|---|---|---|---|---|
| Step time (wall) | 134.4 ms | 171.4 ms | **294.3 ms** | +118.9 % | +71.7 % |
| Device step time | 148.2 ms | 184.9 ms | 318.2 ms | +114.7 % | +72.1 % |
| Tokens/step | 4,096 | 4,096 | **8,192** | +2× | +2× |
| TPS | 30,570 | 23,900 | **27,840** | **−8.9 %** | +16.5 % |
| Peak HBM | 29.69 GiB (95 %) | 21.08 GiB (67 %) | **28.79 GiB (92 %)** | −3 % | +36 % |
| Stack reservation | 17.37 GiB | 8.72 GiB | 16.42 GiB | +0.1 GiB | +7.70 GiB |
| Heap allocation | 12.31 GiB | 12.36 GiB | 12.38 GiB | flat | flat |
| Fragmentation | 0 % | 25.4 % | 0 % | | |
| Per-token cost (µs) | 32.8 | 41.8 | **35.9** | +9.5 % | −14 % |

**Loss descent** (bigger batch benefits loss quality, not relevant for perf comparison but recorded): baseline 3.93 → 1.97, exp 4 3.83 → **1.56**. Expected — batch=2 has lower gradient variance.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp4-batch2-with-remat](http://localhost:8791/?run=2026-04-23-gemma4-exp4-batch2-with-remat) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp4-batch2-with-remat`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp4-batch2-with-remat/`](../../../../../raw/profiles/2026-04-23-gemma4-exp4-batch2-with-remat/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 5, 6, 7
- **What's inside**: xprof trace — full remat + batch=2; HBM at 92 %, per-token cost degraded.

## Mechanism

The doubled batch re-consumed nearly all the memory exp 3 freed. Stack reservation went 17.4 → 8.7 (exp 3) → 16.4 (exp 4). Activations scale linearly with batch, so batch=2 adds ~8 GiB of live activations — which undid the remat headroom almost exactly.

Step time scaled worse than linear: **294.3 / 171.4 = 1.72×** for a 2× token increase. Ideally this would be 1.0× (purely amortized fixed overhead) or at worst 2.0× (pure linear compute). 1.72× implies compute dominates with a small fixed-overhead amortization. The compute itself is still paying remat's +33 % tax; the amortization wins are small because collectives are a minority of time at this size (< 15 %). Per-token cost went 32.8 → 35.9 µs, a net +9.5 % vs baseline.

The chain's per-token cost structure:
- Baseline per-token: compute cost (1×) + collective cost (1× but amortized) = 32.8 µs.
- Exp 3 per-token: 1.33× compute + 2× collective (doubled all-gather) ≈ 41.8 µs.
- Exp 4 per-token: 1.33× compute + 1× collective (amortized across 2× tokens) ≈ 35.9 µs.

In other words: the 2× batch only amortized the **collective** part of the step, not the compute. Compute is 60 %+ of the step at this config, so the amortization recovered ≈ 15 % of the remat tax but not all of it. To break even vs baseline, we'd need batch=4 (further amortization) or a **lower compute tax** (selective remat).

## Verdict

**REFUTED as a throughput win.** The exp 3 + exp 4 chain is net −9 % TPS vs the pre-remat baseline. Correctness preserved (loss descent parallel to baseline, just lower-variance from bigger batch).

The chain is still **informative as a bound**: at this config, full remat + batch=2 is the wrong operating point. batch=4 is not reachable (exp 4 is already at 92 % HBM). The path to actually beating baseline runs through **selective remat** (lower compute tax) — launched as exp 5 in parallel with this writeup.

## Next hypotheses (follow-ups)

1. **Exp 5 (launched)** — selective remat with `jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims` at batch=1. Expected +5–10 % step time, ~50–70 % of exp 3's memory savings. If it works at batch=1, try batch=2 or higher next.
2. **Scan-over-layers** (torchax `ScannedModule` or manual `jax.lax.scan`): collapses 42-layer unroll into one scan-body compile, may share activation buffers across iterations, cuts compile time. Orthogonal to remat.
3. **Host-offload of the single biggest activation** (sibling program.md heuristic 5). More surgical than blanket remat; likely cheaper.
4. **Revisit exp 1 (async-collective flags)** at batch=2 or higher — scheduler needs more compute to hide collectives behind; may flip from regression to win at bigger workload.

## See also

- [Program page](../../README.md), [program.md](../../program.md).
- [OBSERVATIONS.md § exp04](OBSERVATIONS.md).
- [2026-04-23-exp3-full-remat-accepted.md](2026-04-23-exp3-full-remat-accepted.md) — its code change carries into this experiment.
- [rematerialization](../../../../concepts/rematerialization.md), [fsdp](../../../../concepts/fsdp.md), [all-gather](../../../../concepts/all-gather.md).
- [xprof-mcp TPU optimization guide §4.6/§5](../../../../sources/2026-xprof-mcp-tpu-optimization.md).

## Sources

- `raw/profiles/2026-04-23-gemma4-exp4-batch2-with-remat/` — xprof trace.
- `raw/profiles/2026-04-23-gemma4-exp3-full-remat/` — exp 3 counterpart.
- `raw/profiles/2026-04-23-gemma4-loss-confirm/` — baseline counterpart.
- `/tmp/gemma4_exp4.log` — console log.
