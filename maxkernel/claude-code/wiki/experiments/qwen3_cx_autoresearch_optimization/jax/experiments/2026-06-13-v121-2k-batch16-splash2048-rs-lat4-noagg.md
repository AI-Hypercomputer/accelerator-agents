---
title: "Qwen3 JAX current v121 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 no SparseCore aggregator"
type: experiment
hypothesis: "Disabling the SparseCore collective aggregator while keeping all-gather offload disabled may reduce collective scheduling overhead without triggering the all-gather offload OOM path."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, aggregator, reduce-scatter, scheduler, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v121 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 no SparseCore aggregator

## Hypothesis under test

**Hypothesis**: v120 showed all-gather offload is too memory-expensive.
Disabling the SparseCore collective aggregator while keeping all-gather offload
disabled may alter all-reduce/reduce-scatter scheduling with less memory
pressure and potentially improve the v114/v119 graph.

**Mechanism**: Reuse v114 exactly but change:

```bash
--xla_tpu_enable_sparse_core_collective_aggregator=false
```

Keep all-gather offload disabled and all other v114 settings fixed.

## Setup

- Workload: `alekseyv-qwen3-v121-2krsl4noagg`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v121-2k-bs16-vmem98304-splash2048-rslat4-noagg`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v121_2k_bs16_vmem98304_splash2048_rslat4_noagg`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker0 summary: 71,035 tok/s global, 8,879 tok/s/chip, approx 45.7% MFU.
- Worker1 summary: 71,044 tok/s global, 8,881 tok/s/chip, approx 45.7% MFU.
- Profile: `2026_06_13_09_14_12`.
- XProf summary: 3720.7 ms average step time, 67.9% MXU utilization.
- XProf op profile: 89,167.044 ms total, 194.934 ms idle, 0.2% idle.
- Runtime memory: 31.14 GiB / 31.25 GiB peak, with 0.1016 GiB free.
- Optimized train-step HLO:
  `b03d411afe24c9ff0ed55829979aa76f3b9d3b8de4932139e8d398f29a54eff3`,
  19,351,131 bytes / 115,235 lines.
- HLO counts: `all-gather=7063`, `all-reduce=165`,
  `reduce-scatter=2774`, `async-start=373`, `async-done=373`,
  `convolution=3071`, `custom-call=3828`, `copy=17285`,
  `fusion=30238`, `dot_general=7288`, `tokamax=1`.
- Compiled memory: 27.88 GiB total bytes used, with 22.12 GiB
  preallocated temp.

## Verdict

Refuted as an inert flag for this graph. Disabling the SparseCore collective
aggregator produced a byte-identical optimized train-step HLO to v114/v119 and
landed slightly below the v119 measurement best. Keep the v114/v119 graph
settings; do not spend more trials on the aggregator flag unless another
mechanism makes it graph-active.
