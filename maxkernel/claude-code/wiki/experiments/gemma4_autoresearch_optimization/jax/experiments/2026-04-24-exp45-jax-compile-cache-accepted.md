---
title: "Exp 45 — JAX persistent compile cache (ACCEPTED, 6.67× faster total runtime on cache hit)"
type: experiment
tags: [experiment, gemma4, jax, compile-cache, iteration-speed, infra]
hypothesis: jax-compile-cache-accelerates-iteration
model: gemma4-e4b-torchax-jax
created: 2026-04-24
updated: 2026-04-24
commit: pending
verdict: supported
---

Set `JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache` and ran the exp 36 config twice. **First run populates the cache, second run hits it: step 0 compile drops from ~180 s → 14.58 s (12.3×), step 1 recompile drops from ~179 s → 13.84 s (12.9×). Total wall clock 389.4 s → 58.4 s (6.67× faster).** Steady-state TPS unchanged at ~34,600. Pure iteration-speed win, not TPS. Recommending default-on for future experiments since many (esp. parameter sweeps) reuse the same jit shape.

## Hypothesis

Each experiment recompiles the same traced jitted_step at step 0 (~180 s) and step 1 (~179 s, a known recompile pattern from exp 36 observations). A persistent cache shared across runs should make subsequent runs near-instant at the compile stage.

## Setup

Env: `JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache`. Identical stack to exp 36 (splash + b=3 + bf16 CE). Ran twice back-to-back, same command.

## Results

| Metric | First run (cold cache) | Second run (cache hit) | Δ |
|---|---:|---:|---:|
| step 0 compile (s) | 179.82 | **14.58** | −165.24 s (−92 %) |
| step 1 recompile (s) | 179.43 | **13.84** | −165.59 s (−92 %) |
| Steady-state step time (mean, steps 2–19) | ~354 ms | ~354 ms | flat |
| **TPS (steady-state)** | 34,621 | 34,621 | flat |
| **Wall clock (total, 20 steps)** | **389.4 s** | **58.4 s** | **−331 s (6.67× faster)** |
| Loss step 19 | 1.8314 | 1.8314 | bit-identical |

## Profile

- **xprof browser URL**: [2026-04-24-gemma4-jax-exp45-compile-cache-second-run](http://localhost:8791/?run=2026-04-24-gemma4-jax-exp45-compile-cache-second-run) — the cache-hit run's trace.
- **First run**: [`raw/profiles/2026-04-24-gemma4-jax-exp45-compile-cache-first-run/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp45-compile-cache-first-run/)
- **Second run**: [`raw/profiles/2026-04-24-gemma4-jax-exp45-compile-cache-second-run/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp45-compile-cache-second-run/)
- Steady-state TPS trace is indistinguishable from exp 36 — cache only affects compile stage, not runtime.

## Mechanism

`JAX_COMPILATION_CACHE_DIR` tells the PJRT compiler to persist HLO-keyed compiled binaries. On second run with the same jitted shape (trunk unchanged, same args, same sharding), the compiler short-circuits: loads from cache in ~14 s (decompress + validate) instead of retracing + optimizing + lowering (~180 s).

Step 1's "recompile" (a long-open issue in this program — probably from donated-input / output-sharding layout differences at step 1 vs step 0) also gets cached: second run's step 1 is the same 14 s as step 0 after warmup.

## Verdict

**ACCEPTED.** Pure infra win, no TPS change. Recommend setting `JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache` (or a GCS path like `gs://…/jax-cache/` for cross-machine) as default for all future JAX-stack experiments. This compounds with exp 48's splash parameter sweep — any sweep variant that reuses an identical jit shape (block-size changes that don't alter the HLO hash) gets the fast path.

## Next hypotheses enabled by this

- **exp 48 — splash param sweep** runs ~7× faster per variant → 16+ variants testable in the time of 2-3 fresh runs previously. This is the experiment that most benefits from this win.
- **exp 49 — scan-over-layers** would produce a fundamentally new jit shape (cache miss) on first run but then be cached for subsequent tests.

## See also

- [exp 36 — JAX best (the config this tests cache behavior against)](2026-04-23-exp36-jax-splash-batch3-accepted.md)
- [torchax exp 2 — pinned out_shardings to fix step-1 recompile (crashed)](../../torchax/experiments/2026-04-23-exp2-pin-out-shardings-rejected.md) — another angle on the same step-1 recompile issue; now superseded by the compile cache at cost-of-first-run.

## Sources

- `/tmp/gemma4_jax_exp45_first.log` + `/tmp/gemma4_jax_exp45_second.log`
- `raw/profiles/2026-04-24-gemma4-jax-exp45-compile-cache-{first,second}-run/` — cache first/second runs.
