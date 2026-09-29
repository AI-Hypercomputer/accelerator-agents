---
title: "Qwen3 JAX current v125 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 no all-gather ACF"
type: experiment
hypothesis: "Disabling only all-gather async collective fusion may reduce the fused all-gather/convolution schedule pressure on the current 2k graph."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, async-collective-fusion, all-gather, reduce-scatter, scheduler, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v125 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 no all-gather ACF

## Hypothesis under test

**Hypothesis**: The current 2k frontier has all-gathers fused inside the
dominant convolution-fusion bucket. Historical longer-seq all-gather ACF
ablations regressed, but the current BKV2048/RS-lat4 graph has not isolated
only this subtype. Disabling all-gather ACF may reduce schedule pressure or
improve memory packing enough to offset explicit all-gather cost.

**Mechanism**: Reuse the v114/v119 graph settings but change:

```bash
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
```

Keep base async collective fusion enabled, reduce-scatter fusion enabled,
multiple-step ACF enabled, all-reduce ACF disabled, VMEM 98304, shmem100,
reduce-scatter latency multiplier 4, all-reduce latency multiplier 2,
SparseCore all-gather offload disabled, aggregator enabled, batch 16, sequence
2048, and Splash BKV/BKV_COMPUTE 2048 fixed.

## Setup

- Workload: `alekseyv-qwen3-v125-2knoacfg`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v125-2k-bs16-splash2048-rslat4-noacfg`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v125_2k_bs16_splash2048_rslat4_noacfg`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker0 summary: 69,485 tok/s global, 8,686 tok/s/chip, approx 44.7% MFU.
- Worker1 summary: 69,482 tok/s global, 8,685 tok/s/chip, approx 44.7% MFU.
- Profile: `2026_06_13_10_12_32`.
- XProf summary: 3798.8 ms average step time, 66.1% MXU utilization.
- XProf op profile: 91,048.080 ms total, 173.802 ms idle, 0.2% idle.
- Runtime memory: 30.55 GiB / 31.25 GiB peak, with 0.6985 GiB free and
  0.28% fragmentation.
- Optimized train-step HLO:
  `081ebafd20724172c5037e15c8977080cf21b68d680098b342240d8a0e95982b`,
  13,005,762 bytes / 76,725 lines.
- HLO counts: `all-gather=2694`, `all-reduce=165`,
  `reduce-scatter=2774`, `async-start=373`, `async-done=373`,
  `convolution=2891`, `custom-call=1223`, `copy=16336`,
  `fusion=16841`, `dot_general=4624`, `tokamax=1`.
- Compiled memory: 27.32 GiB total bytes used, with 21.57 GiB
  preallocated temp.

## Verdict

Refuted. Disabling all-gather ACF buys substantial memory headroom and shrinks
the HLO, but it exposes a slower schedule: step time regresses by about 80 ms,
MXU falls to 66.1%, and throughput drops to 44.7% MFU. Keep all-gather ACF
enabled on the v114/v119 graph.
