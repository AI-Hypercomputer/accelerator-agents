---
title: "Qwen3 JAX current v117 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 all-reduce latency 3"
type: experiment
hypothesis: "Raising SparseCore all-reduce latency priority from 2 to 3 on the v114 reduce-scatter latency 4 frontier may improve overlap, while v116 showed priority 1 is too low."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, all-reduce, reduce-scatter, scheduler, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v117 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 all-reduce latency 3

## Hypothesis under test

**Hypothesis**: v116 showed all-reduce latency multiplier 1 is too low on the
v114 frontier. Raising the multiplier from the v114 value 2 to 3 may improve
all-reduce overlap enough to beat v114, if it does not cross the tight runtime
memory boundary.

**Mechanism**: Reuse v114 exactly but change:

```bash
--xla_tpu_sparse_core_all_reduce_latency_multiplier=3
```

Keep `--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=4` and all other
v114 settings fixed.

## Setup

- Workload: `alekseyv-qwen3-v117-2krsl4ar3`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v117-2k-bs16-vmem98304-splash2048-rslat4-arlat3`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v117_2k_bs16_vmem98304_splash2048_rslat4_arlat3`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker0 summary: 70,978 tok/s global, 8,872 tok/s/chip, approx 45.6% MFU.
- Worker1 summary: 70,989 tok/s global, 8,874 tok/s/chip, approx 45.6% MFU.
- Profile: `2026_06_13_08_23_46`.
- XProf summary: 3717.8 ms average step time, 67.9% MXU utilization.
- XProf op profile: 89,139.936 ms total, 203.604 ms idle, 0.2% idle.
- Runtime memory: 30.92 GiB / 31.25 GiB peak, with 0.3278 GiB free and
  0.60% fragmentation.
- Optimized train-step HLO:
  `50046481b1b469d9ac0e642de7648eb4bdea368996f787f4224ba64a90fd9ab3`,
  19,353,411 bytes / 115,256 lines.
- HLO counts: `all-gather=7063`, `all-reduce=165`,
  `reduce-scatter=2774`, `async-start=373`, `async-done=373`,
  `convolution=3071`, `custom-call=3828`, `copy=17338`,
  `fusion=30238`, `dot_general=7288`, `tokamax=1`.
- Compiled memory: 27.65 GiB total bytes used, with 21.90 GiB
  preallocated temp.

## Verdict

Not promoted as a throughput frontier: the harness average remains below v114.
However, the profile step is slightly better and runtime memory headroom is much
better than v114, so this is a useful enabling schedule. Next test should retry
reduce-scatter latency multiplier 5 with all-reduce latency multiplier 3 to see
whether v117's memory relief avoids v115's runtime-load OOM.
