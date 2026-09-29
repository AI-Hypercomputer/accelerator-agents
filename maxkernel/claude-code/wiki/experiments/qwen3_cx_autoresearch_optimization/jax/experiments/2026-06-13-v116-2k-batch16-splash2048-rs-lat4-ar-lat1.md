---
title: "Qwen3 JAX current v116 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 all-reduce latency 1"
type: experiment
hypothesis: "Lowering SparseCore all-reduce latency priority from 2 to 1 on the v114 reduce-scatter latency 4 frontier may reduce scheduling pressure without losing useful overlap."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, all-reduce, reduce-scatter, scheduler, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v116 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 all-reduce latency 1

## Hypothesis under test

**Hypothesis**: v114's reduce-scatter latency multiplier 4 is the tentative
frontier, while v115 showed additional reduce-scatter priority crosses the
runtime memory boundary. Lowering
`--xla_tpu_sparse_core_all_reduce_latency_multiplier` from 2 to 1 may reduce
all-reduce scheduling pressure and recover headroom without harming the useful
reduce-scatter hiding.

**Mechanism**: Reuse v114 exactly but change:

```bash
--xla_tpu_sparse_core_all_reduce_latency_multiplier=1
```

Keep `--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=4` and all other
v114 settings fixed.

## Setup

- Workload: `alekseyv-qwen3-v116-2krsl4ar1`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v116-2k-bs16-vmem98304-splash2048-rslat4-arlat1`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v116_2k_bs16_vmem98304_splash2048_rslat4_arlat1`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker0 summary: 70,960 tok/s global, 8,870 tok/s/chip, approx 45.6% MFU.
- Worker1 summary: 70,952 tok/s global, 8,869 tok/s/chip, approx 45.6% MFU.
- Profile: `2026_06_13_08_09_47`.
- XProf summary: 3726.8 ms average step time, 67.7% MXU utilization.
- XProf op profile: 89,281.391 ms total, 190.618 ms idle, 0.2% idle.
- Runtime memory: 31.15 GiB / 31.25 GiB peak, with 0.1008 GiB free and
  1.94% fragmentation.
- Optimized train-step HLO:
  `ce008beb94648c98bd92a7bacb2f70a5ce8d00a633f1e4540be0a4a4b180d1a8`,
  19,354,349 bytes / 115,268 lines.
- HLO counts: `all-gather=7063`, `all-reduce=165`,
  `reduce-scatter=2774`, `async-start=376`, `async-done=376`,
  `convolution=3071`, `custom-call=3831`, `copy=17302`,
  `fusion=30238`, `dot_general=7288`, `tokamax=1`.
- Compiled memory: 27.88 GiB total bytes used, with 22.12 GiB
  preallocated temp.

## Verdict

Refuted. Lowering the all-reduce latency multiplier preserves the memory
envelope but regresses throughput and profile step time relative to v114. Do not
carry all-reduce latency multiplier 1.
