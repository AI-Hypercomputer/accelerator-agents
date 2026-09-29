---
title: "Qwen3 JAX current v126 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 no ACF multiple-steps"
type: experiment
hypothesis: "Disabling only multiple-step async collective fusion may reduce scheduling pressure while preserving all-gather and reduce-scatter fusion."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, async-collective-fusion, multiple-steps, reduce-scatter, scheduler, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v126 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 no ACF multiple-steps

## Hypothesis under test

**Hypothesis**: v125 showed all-gather ACF itself is required for throughput,
but it also revealed that ACF subtype changes can free memory. Disabling only
multiple-step ACF may reduce scheduling pressure while preserving the important
all-gather and reduce-scatter fusion paths.

**Mechanism**: Reuse the v114/v119 graph settings but change:

```bash
--xla_tpu_enable_async_collective_fusion_multiple_steps=false
```

Keep base async collective fusion enabled, all-gather fusion enabled,
reduce-scatter fusion enabled, all-reduce ACF disabled, VMEM 98304, shmem100,
reduce-scatter latency multiplier 4, all-reduce latency multiplier 2,
SparseCore all-gather offload disabled, aggregator enabled, batch 16, sequence
2048, and Splash BKV/BKV_COMPUTE 2048 fixed.

## Setup

- Workload: `alekseyv-qwen3-v126-2knoacfms`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v126-2k-bs16-splash2048-rslat4-noacfms`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v126_2k_bs16_splash2048_rslat4_noacfms`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker 0: 70,838 tok/s global, 8,855 tok/s/chip, approx 45.5% MFU.
- Worker 1: 70,858 tok/s global, 8,857 tok/s/chip, approx 45.6% MFU.
- XProf run: `2026_06_13_10_25_56`.
- XProf profile summary: 3727.6 ms average step time, 67.8% MXU utilization,
  0.3% idle in op profile.
- XProf memory profile: 31.05 GiB peak / 31.25 GiB capacity, 0.1948 GiB free,
  99.36% utilization.
- HLO SHA256:
  `4c515807306666ee46948268f8732ca0362e6a729e2d96fea81424de6ee33518`.
- HLO size: 13,345,898 bytes / 76,638 lines.
- HLO op-token counts: all-gather=2364, all-reduce=165,
  reduce-scatter=2774, async-start=253, async-done=253, convolution=2981,
  custom-call=1124, copy=13330, fusion=17405, dot_general=5631, tokamax=1.
- Compiled memory report: 27.85 GiB total, 22.10 GiB temp.

## Verdict

Refuted. Disabling only ACF multiple-steps saves graph size and some memory
relative to the v119 frontier graph, but it also drops throughput to 70,858
tok/s and lengthens profile step time to 3727.6 ms. The frontier still needs
full async collective fusion multiple-steps enabled.
