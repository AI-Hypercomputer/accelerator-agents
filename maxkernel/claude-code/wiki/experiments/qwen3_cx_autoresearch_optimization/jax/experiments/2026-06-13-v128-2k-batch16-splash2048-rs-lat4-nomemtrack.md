---
title: "Qwen3 JAX current v128 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 no TPU scheduler memory-pressure tracking"
type: experiment
hypothesis: "Disabling TPU scheduler memory-pressure tracking may reduce scheduling overhead or preserve the frontier HLO with no throughput loss on the 2k batch16 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, scheduler, memory-pressure-tracking, collectives, async-collective-fusion, reduce-scatter, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v128 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 no TPU scheduler memory-pressure tracking

## Hypothesis under test

**Hypothesis**: The current 2k frontier is HBM-tight but not compile-OOMing.
Disabling TPU scheduler memory-pressure tracking may reduce scheduler overhead
or keep the same final graph with no runtime penalty.

**Mechanism**: Start from the v119 named 2k frontier and add only:

```bash
--xla_tpu_enable_scheduler_memory_pressure_tracking=false
```

Keep batch 16, sequence 2048, Splash BKV/BKV_COMPUTE 2048, VMEM 98304,
shmem100, scheduler rerun 4, async collective fusion enabled, all-gather and
reduce-scatter ACF enabled, all-reduce ACF disabled, RS latency multiplier 4,
AR latency multiplier 2, SparseCore all-gather offload disabled,
reduce-scatter/all-reduce offload enabled, and aggregator enabled.

## Setup

- Workload: `alekseyv-qwen3-v128-2knomem`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v128-2k-bs16-splash2048-rslat4-nomem`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v128_2k_bs16_splash2048_rslat4_nomem`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker 0: 70,985 tok/s global, 8,873 tok/s/chip, approx 45.6% MFU.
- Worker 1: 70,983 tok/s global, 8,873 tok/s/chip, approx 45.6% MFU.
- XProf run: `2026_06_13_10_53_48`.
- XProf profile summary: 3723.8 ms average step time, 67.9% MXU utilization,
  0.2% idle in op profile.
- XProf memory profile: 31.14 GiB peak / 31.25 GiB capacity, 0.1016 GiB free,
  99.65% utilization.
- HLO SHA256:
  `b03d411afe24c9ff0ed55829979aa76f3b9d3b8de4932139e8d398f29a54eff3`.
- HLO size: 19,351,131 bytes / 115,235 lines.
- HLO op-token counts: all-gather=7063, all-reduce=165,
  reduce-scatter=2774, async-start=373, async-done=373, convolution=3071,
  custom-call=3828, copy=17285, fusion=30238, dot_general=7288, tokamax=1.
- Compiled memory report: 27.88 GiB total, 22.12 GiB temp.
- The optimized HLO hash is byte-identical to v114/v119/v121/v122.

## Verdict

Refuted/tie. Disabling TPU scheduler memory-pressure tracking leaves the
frontier HLO unchanged and does not improve throughput or profile step time.
Keep the default memory-pressure tracking behavior and stop treating this
scalar flag as a useful 2k lever.
