---
title: "Exp 42 — JAX async_collective_fusion in isolation (POTENTIAL, +0.04% flat)"
type: experiment
tags: [experiment, gemma4, jax, xla-flags, collective-fusion, culprit-isolation]
hypothesis: async-collective-fusion-alone-is-innocent
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: pending
verdict: inconclusive
---

Enabled only `--xla_tpu_enable_async_collective_fusion=true` via `LIBTPU_INIT_ARGS` on the JAX stack at exp 36's config. **Result: 34,629 TPS vs exp 36's 34,614 — +0.043 % flat.** A third data point in the culprit-isolation series alongside torchax exp 30 / 31 and JAX exp 38: none of the individually-named collective-overlap flags regress; the bundle regression in torchax exp 1 / 13 / 21 must come from specific **combinations** (likely `fuse_all_gather` + `multiple_steps` together).

## Hypothesis

Each individual `xla_tpu_enable_async_collective_fusion*` flag might be a no-op on its own; the bundle's −20 % regression (torchax exp 1) may be a combinatorial effect. Isolate this base flag.

## Results

| Metric | Exp 36 (default) | **Exp 42 (+async_collective_fusion)** | Δ |
|---|---|---|---|
| TPS | 34,614 | **34,629** | **+0.04 %** |
| Step time (mean, steps 2–19) | 355.0 ms | **354.85 ms** | −0.04 % |
| MFU | 23.05 % | **23.06 %** | +0.01 pt |
| Compile step 0 | 132 s | 165.91 s | +26 % |
| Loss descent | clean | match | identical (step 19 = 1.83 both) |

Compile bumped (scheduler has more search space) but runtime identical within noise.

## Meta-finding: complete collective-flag isolation matrix

Combined with prior isolation tests on BOTH stacks:

| Flag | Torchax result | JAX result |
|---|---|---|
| `latency_hiding_scheduler` | flat (exp 30, −0.003 %) | flat (exp 38, +0.06 %) |
| `overlap_compute_collective_tc` | flat (exp 31, −0.13 %) | not tested (expected flat) |
| `async_collective_fusion` | not tested in isolation | **flat (exp 42, +0.04 %)** |
| `async_collective_fusion_fuse_all_gather` | not tested in isolation | not tested |
| `async_collective_fusion_multiple_steps` | not tested in isolation | not tested |
| **Full bundle** (all above) | **−20 % (torchax exp 1)** + **−12 % (torchax exp 13, on splash)** | not tested on JAX |

Three of five flags now confirmed flat in isolation, on one stack each. **The bundle regression is combinatorial** — probably `fuse_all_gather=true` + `multiple_steps=true` together cause the scheduler to produce a schedule that over-commits compute locality. A further exp could isolate those two, but the bundle is permanently parked on torchax anyway (diminishing returns).

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-jax-exp42-async-collective-fusion](http://localhost:8791/?run=2026-04-23-gemma4-jax-exp42-async-collective-fusion) — opens the trace viewer.
- **Run name**: `2026-04-23-gemma4-jax-exp42-async-collective-fusion`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-jax-exp42-async-collective-fusion/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp42-async-collective-fusion/) (gitignored; GCS mirror).
- **Steps captured**: 10, 11, 12.

## Verdict

**POTENTIAL / INCONCLUSIVE.** Flat. Not merged (no value, +26 % compile cost).

## Next hypotheses

- **Possibly done with XLA-flag isolation on this stack.** Remaining untested flags (`fuse_all_gather`, `multiple_steps`) are expected to combine badly with `async_collective_fusion` per the bundle evidence; skip.
- **JAX-stack ceiling analysis**: 7 experiments since exp 36 (exp 37–42) produced zero further TPS gains. This is the same pattern as the torchax arc (exp 26–33 after exp 25 ceiling). File a second ceiling-analysis page covering the JAX stack, the novel JAX-specific findings (port bugs fixed, lower compile-time HBM), and the cumulative torchax+JAX lesson set.
- **Scan-over-layers** (compile-time, not TPS). Remaining structural lever. JAX port can implement cleanly. ~300-500 LOC.

## See also

- [exp 36 — JAX best](2026-04-23-exp36-jax-splash-batch3-accepted.md)
- [exp 38 — latency-hiding alone flat](2026-04-23-exp38-jax-latency-hiding-potential.md)
- [torchax exp 30 — latency-hiding alone flat](../../torchax/experiments/2026-04-23-exp30-latency-hiding-solo-potential.md)
- [torchax exp 31 — overlap_compute_collective_tc alone flat](../../torchax/experiments/2026-04-23-exp31-overlap-compute-collective-tc-potential.md)
- [torchax exp 1 — full bundle refuted −20 %](../../torchax/experiments/2026-04-23-exp1-async-collective-flags-rejected.md)

## Sources

- `/tmp/gemma4_jax_exp42.log`
- `raw/profiles/2026-04-23-gemma4-jax-exp42-async-collective-fusion/` — xprof run `2026-04-23-gemma4-jax-exp42-async-collective-fusion` at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp42-async-collective-fusion
