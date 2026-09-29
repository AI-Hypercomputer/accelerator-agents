---
title: "Qwen3 JAX current v127 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 no base ACF"
type: experiment
hypothesis: "Disabling base async collective fusion on the current 2k frontier graph will expose whether the all-gather ACF regression from v125 is caused by the base gate or the all-gather subtype specifically."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, async-collective-fusion, reduce-scatter, scheduler, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v127 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 no base ACF

## Hypothesis under test

**Hypothesis**: v125 showed disabling only all-gather ACF regresses badly, while
v126 showed disabling only ACF multiple-steps also regresses. Disabling the base
ACF gate should reproduce the all-gather-ACF regression if the base gate is
effectively required for the same schedule family.

**Mechanism**: Start from the v119 named 2k frontier and change only:

```bash
--xla_tpu_enable_async_collective_fusion=false
```

Keep all subtype flags at the frontier values, scheduler rerun 4, VMEM 98304,
shmem100, reduce-scatter latency multiplier 4, all-reduce latency multiplier 2,
SparseCore all-gather offload disabled, reduce-scatter/all-reduce offload
enabled, aggregator enabled, batch 16, sequence 2048, and Splash BKV/BKV_COMPUTE
2048.

## Setup

- Workload: `alekseyv-qwen3-v127-2knoacf`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v127-2k-bs16-splash2048-rslat4-noacf`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v127_2k_bs16_splash2048_rslat4_noacf`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker 0: 69,472 tok/s global, 8,684 tok/s/chip, approx 44.7% MFU.
- Worker 1: 69,470 tok/s global, 8,684 tok/s/chip, approx 44.7% MFU.
- XProf run: `2026_06_13_10_41_23`.
- XProf profile summary: 3803.5 ms average step time, 66.1% MXU utilization,
  0.2% idle in op profile.
- XProf memory profile: 30.55 GiB peak / 31.25 GiB capacity, 0.6985 GiB free,
  97.76% utilization.
- HLO SHA256:
  `081ebafd20724172c5037e15c8977080cf21b68d680098b342240d8a0e95982b`.
- HLO size: 13,005,762 bytes / 76,725 lines.
- HLO op-token counts: all-gather=2694, all-reduce=165,
  reduce-scatter=2774, async-start=373, async-done=373, convolution=2891,
  custom-call=1223, copy=16336, fusion=16841, dot_general=4624, tokamax=1.
- Compiled memory report: 27.32 GiB total, 21.57 GiB temp.
- The optimized HLO hash matches v125 exactly.

## Verdict

Refuted. Disabling base ACF collapses to the same slower schedule family as
v125's no-all-gather-ACF experiment, with substantially lower throughput and
profile quality than the v119 frontier. Keep base async collective fusion
enabled.
