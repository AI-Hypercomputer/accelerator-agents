---
title: "Qwen3 JAX current v112 - 2k batch16 Splash BKV2048 no SparseCore all-reduce offload"
type: experiment
hypothesis: "Disabling SparseCore all-reduce offload may reduce scheduling pressure from the small all-reduce set while preserving the successful BKV2048 attention and reduce-scatter schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, all-reduce, offload, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v112 - 2k batch16 Splash BKV2048 no SparseCore all-reduce offload

## Hypothesis under test

**Hypothesis**: The current BKV2048 frontier keeps SparseCore all-reduce
offload enabled, but the train-step only has **165** all-reduce references.
The v111 concurrency cap was HLO-inert, so the remaining useful one-variable
probe is to remove the SparseCore all-reduce offload itself while keeping
SparseCore reduce-scatter offload and all-gather ACF unchanged.

**Mechanism**: Reuse v095/v096 exactly but change:

```bash
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false
```

Keep `--batch_size=16`, `--seqlen=2048`, scoped VMEM **98304 KiB**,
`--xla_tpu_scheduler_percent_shared_memory_limit=100`, no-scan remat,
activation sharding, Tokamax Splash, MaxText CE, SparseCore reduce-scatter
offload, SparseCore collective aggregator, all-gather ACF enabled,
reduce-scatter ACF enabled, all-reduce ACF disabled, multiple-step ACF enabled,
and:

```bash
export SPLASH_BKV=2048
export SPLASH_BKV_COMPUTE=2048
```

**Predicted signal**: Support requires beating the confirmed v095/v096
**70,959-70,976 tok/s** band or matching it with lower XProf step time. A
same-HLO or slower changed-HLO runtime refutes disabling all-reduce offload.

## Setup

- Workload: `alekseyv-qwen3-jax-v112-2ksp2048noaroff`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v112-2k-bs16-vmem98304-splash2048-noaroff`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v112_2k_bs16_vmem98304_splash2048_noaroff`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 16 / global batch 128; sequence length
  **2048**; profile window steps 12-14.

## Status

Completed on 2026-06-13.

## Results

- Workload `alekseyv-qwen3-jax-v112-2ksp2048noaroff` completed cleanly with
  `EXIT_CODE=0`.
- Worker throughput:
  - worker0: **70,795 tok/s global**, **8,849 tok/s/chip**, **45.5% MFU**.
  - worker1: **70,804 tok/s global**, **8,851 tok/s/chip**, **45.5% MFU**.
- XProf run queried with xprof pointed directly at the v112 run directory:
  `2026_06_13_07_14_38`.
- XProf summary: **3730.0 ms** step time, **67.5%** MXU utilization,
  **0.2%** idle in the op profile.
- Runtime memory improved but did not help throughput: **30.60 GiB / 31.25 GiB**
  peak use, **0.6493 GiB** free, **22.7457 GiB** stack reservation,
  **7.8511 GiB** heap, **0.3%** fragmentation.
- Op profile remains dominated by dense work: convolution fusion
  **60,038.191 ms / 67.1%**, total train-step program time
  **89,250.728 ms**, idle **216.667 ms / 0.2%**.
- Optimized train-step HLO changed:
  - SHA256:
    `557078060d617d27e02f09218d4a1c240d9ac4e16ff60824bb3919a5877b55ff`.
  - Size: **19,386,544 bytes / 115,563 lines**.
  - Counts: `all-gather=7069`, `all-reduce=156`,
    `reduce-scatter=2774`, `async-start=387`, `async-done=387`,
    `convolution=3071`, `custom-call=3837`, `copy=17906`,
    `fusion=30254`, `dot_general=7298`, `tokamax=1`.
  - Compiled memory report: **28.49 GiB** total, **22.74 GiB**
    preallocated temp.

## Verdict

Refuted. Disabling SparseCore all-reduce offload changes the HLO and frees
runtime memory, but it increases copies, lowers MXU utilization, and regresses
throughput. Keep SparseCore all-reduce offload enabled.
