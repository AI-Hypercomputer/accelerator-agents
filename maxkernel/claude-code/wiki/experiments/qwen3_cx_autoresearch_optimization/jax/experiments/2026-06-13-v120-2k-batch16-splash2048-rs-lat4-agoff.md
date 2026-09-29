---
title: "Qwen3 JAX current v120 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 all-gather offload"
type: experiment
hypothesis: "Enabling SparseCore all-gather collective offload with the current aggregator may reduce all-gather pressure in the dominant convolution fusions."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: failed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, all-gather, offload, reduce-scatter, scheduler, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v120 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 all-gather offload

## Hypothesis under test

**Hypothesis**: The dominant profile remains convolution fusions with embedded
all-gathers. Enabling SparseCore all-gather collective offload while keeping
the SparseCore collective aggregator enabled may improve overlap or reduce
pressure compared with the v114/v119 graph.

**Mechanism**: Reuse v114 exactly but change:

```bash
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
```

Keep all other v114 settings fixed, including reduce-scatter latency 4,
all-reduce latency 2, SparseCore all-reduce/reduce-scatter offload, and
aggregator enabled.

## Setup

- Workload: `alekseyv-qwen3-v120-2krsl4agoff`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v120-2k-bs16-vmem98304-splash2048-rslat4-agoff`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v120_2k_bs16_vmem98304_splash2048_rslat4_agoff`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Workload failed before step 0 with `RESOURCE_EXHAUSTED` while loading
  `jit_train_step`.
- Runtime error: attempting to allocate 2.23 GiB with only 1.60 GiB free.
- Optimized train-step HLO:
  `0283d250de511c60f43462edf47712f5996549a9130b9d562cd3108b729d2a01`,
  16,475,320 bytes / 94,949 lines.
- HLO counts: `all-gather=8391`, `all-reduce=165`,
  `reduce-scatter=2774`, `async-start=1426`, `async-done=1426`,
  `convolution=2901`, `custom-call=4187`, `copy=22253`,
  `fusion=16773`, `dot_general=6363`, `tokamax=1`.
- Compiled memory: 28.50 GiB total bytes used, with 22.75 GiB
  preallocated temp.

## Verdict

Hard refuted. SparseCore all-gather offload creates a much more async-heavy and
copy-heavy HLO, increases compiled memory, and fails program load. Keep
SparseCore all-gather offload disabled on this frontier.
