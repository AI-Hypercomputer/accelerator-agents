---
title: "Qwen3 JAX current v118 - 2k batch16 Splash BKV2048 reduce-scatter latency 5 all-reduce latency 3"
type: experiment
hypothesis: "Combining reduce-scatter latency multiplier 5 with all-reduce latency multiplier 3 may recover v115's desired reduce-scatter priority while using v117's lower memory schedule to avoid runtime load OOM."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: failed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, all-reduce, reduce-scatter, scheduler, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v118 - 2k batch16 Splash BKV2048 reduce-scatter latency 5 all-reduce latency 3

## Hypothesis under test

**Hypothesis**: v115's reduce-scatter latency multiplier 5 failed at program
load due to memory pressure. v117's all-reduce latency multiplier 3 reduced
compiled and runtime memory but did not improve harness throughput. Combining
RS latency 5 with AR latency 3 may make the stronger reduce-scatter hiding
schedule fit and potentially beat v114.

**Mechanism**: Reuse v117 exactly but change:

```bash
--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=5
```

Keep `--xla_tpu_sparse_core_all_reduce_latency_multiplier=3` and all other
settings fixed.

## Setup

- Workload: `alekseyv-qwen3-v118-2krsl5ar3`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v118-2k-bs16-vmem98304-splash2048-rslat5-arlat3`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v118_2k_bs16_vmem98304_splash2048_rslat5_arlat3`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Workload failed before step 0 with `RESOURCE_EXHAUSTED` while loading
  `jit_train_step`.
- Runtime error: attempting to allocate 2.12 GiB with only 1.86 GiB free.
- Optimized train-step HLO:
  `d4897181619cde937837186d2856d72e08a9af32e11930f41e5824f5412c8e53`,
  19,345,317 bytes / 115,222 lines.
- HLO counts: `all-gather=7051`, `all-reduce=165`,
  `reduce-scatter=2774`, `async-start=373`, `async-done=373`,
  `convolution=3078`, `custom-call=3828`, `copy=17383`,
  `fusion=30190`, `dot_general=7280`, `tokamax=1`.
- Compiled memory: 28.25 GiB total bytes used, with 22.50 GiB
  preallocated temp.

## Verdict

Hard refuted. Combining AR latency 3 with RS latency 5 does not preserve
v117's memory relief; the compiled memory and runtime load failure match v115's
bad RS5 shape. The reduce-scatter latency multiplier 5 path is closed for this
frontier.
