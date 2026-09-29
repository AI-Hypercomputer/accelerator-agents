---
title: "Exp 1 — async-collective XLA flags (REFUTED)"
type: experiment
tags: [experiment, gemma4, xla-flags, async-collectives, refuted]
hypothesis: async-collective-xla-flags
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "wiki:HEAD (torchax submodule 8f957d1); train.py post-baseline fixes"
verdict: refuted
---

Tests the standard **async-collective-fusion + latency-hiding-scheduler** XLA flag bundle on the Gemma 4 E4B torchax FSDP baseline. The [2026-04-22 baseline](2026-04-22-baseline.md) showed ~10–15 % of step time in all-gather / all-reduce-scatter / all-reduce plus ~60 ms of `async-collective-done` wait. The hypothesis was that overlapping these with compute would shave 5–10 % off step time. **It regressed by 25 %.**

## Hypothesis under test

**Statement**: Enabling `--xla_tpu_enable_latency_hiding_scheduler` + `--xla_tpu_enable_async_collective_fusion` + the multi-step / fuse-all-gather variants will reduce per-step wall-clock on Gemma 4 E4B at seq=1024, batch=1, FSDP=4 on v6e-4 by **5–10 %** without changing the loss trajectory.

Originates from [async-collectives](../../../../concepts/async-collectives.md), [latency-hiding-scheduler](../../../../concepts/latency-hiding-scheduler.md), and hypothesis #9 on the [program page](../../README.md).

## Setup

- Hardware: 4 × TPU v6 Lite (same host as baseline).
- Conda env: `gemma4_py313` (unchanged).
- Flags delta vs [2026-04-22 baseline](2026-04-22-baseline.md):
  - `LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=131072 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true"` (5 flags set; baseline set none).
  - One attempted flag `--xla_tpu_overlap_compute_collective_comms=true` was rejected by libtpu 0.0.40 (`Unknown command line flag`); dropped before the final run. **Post-hoc:** the real flag is `--xla_tpu_overlap_compute_collective_tc` (my mental cache had the wrong suffix). Confirmed by symbol-dumping `libtpu.so` (`strings libtpu.so | grep xla_tpu_overlap_compute_`). libtpu 0.0.40 IS the latest version (verified against PyPI, libtpu-lts-releases, libtpu-nightly-releases — nightlies stalled at `0.0.9.dev20250207`). No version upgrade available; only rename the flag.
  - Also tried the same flags in `XLA_FLAGS` first — `F0423 01:13:12.780461 999760 parse_flags_from_env.cc:234] Unknown flags in XLA_FLAGS`. **`--xla_tpu_*` flags go in `LIBTPU_INIT_ARGS`, not `XLA_FLAGS`.**
- Command:

  ```bash
  LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=131072 \
    --xla_tpu_enable_latency_hiding_scheduler=true \
    --xla_tpu_enable_async_collective_fusion=true \
    --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
    --xla_tpu_enable_async_collective_fusion_multiple_steps=true" \
  python -m train --steps 20 --batch_size 1 --seq_len 1024 \
    --profile_dir raw/profiles/2026-04-23-gemma4-exp1-async-collectives \
    --profile_steps 10 11 12
  ```

## Baseline comparison

Baseline: seq=1024 run of [2026-04-22 baseline](2026-04-22-baseline.md) (20-step rerun dated 2026-04-23 at `raw/profiles/2026-04-23-gemma4-loss-confirm/`, identical config minus the LIBTPU flags).

## Results

| Metric | Baseline (seq=1024) | Exp1 (seq=1024 + flags) | Delta | Noise band |
|---|---|---|---|---|
| Steady-state step time (wall-clock) | **134.4 ms** | **168.3 ms** | **+33.9 ms (+25.2 %)** | σ < 1 ms each |
| Device step time (xprof overview) | 148.2 ms | 182.8 ms | +34.6 ms (+23.3 %) | σ = 0.8–0.9 ms |
| Compile step 0 | 149.5 s | 159.4 s | +9.9 s (+6.6 %) | single sample |
| Compile step 1 (recompile) | 150.1 s | 160.4 s | +10.3 s (+6.9 %) | single sample |
| Loss, step 0 | 3.9339 | 3.9291 | –0.0048 | bf16 reorder noise |
| Loss, step 14 (lowest of the run) | 1.9685 | 1.9511 | –0.0174 | noise |
| Tokens/sec | ~30,500 | ~24,300 | **–20 %** | |

**HLO-op-level diff** (3 profiled steps, aggregated across 4 chips; `get_top_hlo_ops`):

| Op | Baseline time (ms) | Baseline bytes (GiB) | Exp1 time (ms) | Exp1 bytes (GiB) | Time Δ | Bytes Δ |
|---|---|---|---|---|---|---|
| convolution fusion | 613 | 253 | 791 | **639** | **+178 ms** | **+2.5×** |
| loop fusion | 321 | 305 | 506 | **568** | **+185 ms** | **+1.9×** |
| custom fusion | 182 | 12 | 235 | 262 | +53 ms | +22× |
| all-gather | 111 | 1.0 | 106 | 15.8 | **–5 ms** | +16× |
| all-reduce-scatter fusion | 75 | 2.0 | 66 | 12.3 | **–9 ms** | +6× |
| all-reduce | 46 | 2.8 | 63 | 6.4 | +17 ms | +2.3× |
| data formatting | 40 | 25.8 | 72 | 65.1 | +32 ms | +2.5× |
| Total profile | 1,694 | 953 | 2,108 | ≈ 1,975 | **+414 ms** | **+1.1 GiB** |

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp1-async-collectives](http://localhost:8791/?run=2026-04-23-gemma4-exp1-async-collectives) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp1-async-collectives`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp1-async-collectives/`](../../../../../raw/profiles/2026-04-23-gemma4-exp1-async-collectives/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — async-collective XLA flag bundle; HLO-op diff vs baseline shows the compute-fusion bytes explosion.

## Observations

1. **The collectives themselves did get slightly faster.** all-gather −5 ms, all-reduce-scatter −9 ms. So the flags did what they advertise at the collective level.
2. **But compute fusions regressed heavily.** convolution fusion +178 ms (+2.5× memory traffic), loop fusion +185 ms (+1.9× memory traffic). The scheduler's reordering broke the compute-side memory locality that the baseline had by default — many more materializations to HBM.
3. **Bytes accessed for every collective went up too** (all-gather 1→16 GiB, all-reduce-scatter 2→12 GiB). The fused variants are communicating more bytes, likely because the scheduler deferred them further from their consumers, forcing larger live tensors.
4. Compile time grew ~7 % (149.5 → 159.4 s per compile). The fused scheduler has more work to do. Not a dominant concern since it's one-time-ish (well, two-time — step-0 + step-1 recompile is still open).
5. **Loss trajectory is identical within bf16-reorder noise.** No correctness change. The verdict is purely performance-refuted, not invalid.
6. **The async-collective-done wait time from the baseline (~64 ms summed across `custom fusion`) was real.** Exp1 did reduce that per-op, but the cost was paid elsewhere. Net loss.

### Likely mechanism

The stock XLA scheduler for this workload produced a good compute-ordered schedule because the workload is small (batch=1, seq=1024), fitting comfortably. Enabling async-collective fusion + latency-hiding gave the scheduler more freedom to reorder; it chose reorderings that fused collectives but pushed convolution/loop fusions apart from their consumers, forcing extra HBM traffic.

This matches the xprof-mcp TPU optimization guide's caveat that these flags are "workload-dependent" — useful at scale, not universally.

## Verdict

**REFUTED.** The hypothesis predicted a 5–10 % wall-clock reduction; the measurement shows a **25 % wall-clock regression** and a **~1 GiB/step increase in total bytes accessed**. Correctness preserved (loss trajectory matches within numerical noise).

Valid as evidence at **seq=1024, batch=1, FSDP=4, v6e-4** only. At larger effective batch (seq ≥ 2048 once NaN is fixed, or batch > 1 once memory permits via remat), the balance between collective-overlap wins and scheduler-reorder losses may flip. Keep the hypothesis parked rather than retired — revisit once **#6 (selective remat)** or **NaN fix** unlocks larger configs.

## Next hypotheses

1. **Fix step-1 recompile** (~150 s/run savings; pure tooling). Mechanical: pass explicit `in_shardings` / `out_shardings` to `jax.jit` in `train.py` so step 1's donation path hits the step-0 cache key. Highest ROI for iteration speed on this program.
2. **Selective rematerialization** (hypothesis #6 on [README](../../README.md)). Memory pressure on baseline was 95 % HBM; remat frees activations, unlocks larger batch, which is where the async-collective flags might actually pay off.
3. **Fix NaN at seq≥2048** (hypothesis #0' / prereq). Cast logits to fp32 before log-softmax; if that doesn't fix it, compare per-layer hidden-state stats to localize the overflow.
4. **Revisit async-collective flags at larger effective batch** once #2 or #3 lands.

## See also

- [Program page](../../README.md) — open hypotheses list, history.
- [2026-04-22 baseline](2026-04-22-baseline.md) — reference numbers.
- [async-collectives](../../../../concepts/async-collectives.md), [latency-hiding-scheduler](../../../../concepts/latency-hiding-scheduler.md), [collective-communication](../../../../concepts/collective-communication.md), [xla-flags](../../../../concepts/xla-flags.md).
- [xprof-mcp TPU optimization guide](../../../../sources/2026-xprof-mcp-tpu-optimization.md) — flag catalog.
- [xprof — HLO op stats](../../../../sources/2026-xprof-hlo-op-stats.md) — how the op-diff table above was read.

## Sources

- `raw/profiles/2026-04-23-gemma4-exp1-async-collectives/` — xprof trace for this experiment.
- `raw/profiles/2026-04-23-gemma4-loss-confirm/` — baseline counterpart.
- `/tmp/gemma4_exp1.log` — full console log of the run (not persisted in the wiki; values extracted into this page).
