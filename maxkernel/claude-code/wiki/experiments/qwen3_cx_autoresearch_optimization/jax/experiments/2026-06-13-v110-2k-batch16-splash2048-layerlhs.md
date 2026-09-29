---
title: "Qwen3 JAX current v110 - 2k batch16 Splash BKV2048 layer scheduler true"
type: experiment
hypothesis: "Forcing the latency-hiding layer scheduler on may improve schedule quality on the BKV2048 frontier where sparse-core collective aggregation is already enabled."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, scheduler, latency-hiding-scheduler, layer-scheduler, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v110 - 2k batch16 Splash BKV2048 layer scheduler true

## Hypothesis under test

**Hypothesis**: The current BKV2048 frontier keeps
`--xla_tpu_enable_latency_hiding_layer_scheduler=auto` while sparse-core
collective aggregation is already enabled. Forcing the layer scheduler on may
change schedule formation around the dense matmul plus offloaded collective
mix and recover a small amount of step time.

**Mechanism**: Reuse v095/v096 exactly and add only:

```bash
--xla_tpu_enable_latency_hiding_layer_scheduler=true
```

Keep `--batch_size=16`, `--seqlen=2048`, scoped VMEM **98304 KiB**,
`--xla_tpu_scheduler_percent_shared_memory_limit=100`, no-scan remat,
activation sharding, Tokamax Splash, MaxText CE, SparseCore reduce-scatter and
all-reduce offload, SparseCore collective aggregator, all-gather ACF enabled,
reduce-scatter ACF enabled, all-reduce ACF disabled, multiple-step ACF enabled,
and:

```bash
export SPLASH_BKV=2048
export SPLASH_BKV_COMPUTE=2048
```

**Predicted signal**: Support requires beating the confirmed v095/v096
**70,959-70,976 tok/s** band or matching it with lower XProf step time. A
same-HLO in-band runtime refutes the explicit layer-scheduler setting.

## Setup

- Workload: `alekseyv-qwen3-jax-v110-2ksp2048layerlhs`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v110-2k-bs16-vmem98304-splash2048-layerlhs`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v110_2k_bs16_vmem98304_splash2048_layerlhs`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 16 / global batch 128; sequence length
  **2048**; profile window steps 12-14.

## Status

Completed on 2026-06-13 with compile-time OOM.

## Results

- Workload `alekseyv-qwen3-jax-v110-2ksp2048layerlhs` failed during
  train-step compilation with `EXIT_CODE=1`.
- Error:
  `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 78.08G of 31.25G hbm. Exceeded hbm capacity by 46.84G.`
- The compiler reported program HBM requirement **78.08 GiB** and HLO temp
  **78.01 GiB**.
- Optimized train-step HLO:
  - SHA256:
    `a93471451b4a2868f507f4eb56fea29378d5e5742242c338bfa3d9934abe21d9`.
  - Size: **15,893,236 bytes / 94,891 lines**.
  - Counts: `all-gather=4928`, `all-reduce=165`,
    `reduce-scatter=2774`, `async-start=253`, `async-done=253`,
    `convolution=3242`, `custom-call=2220`, `copy=22493`,
    `fusion=22211`, `dot_general=5976`, `tokamax=1`.
- Memory report: **83.76 GiB** total bytes used, with **78.01 GiB**
  preallocated temp.
- OOM top-memory file shows the largest repeated temps are 768 MiB
  `bf16[16,2048,12288]` remat dot/fusion allocations fed by all-gather
  producers, plus a 1.16 GiB SparseCore all-reduce temp.

## Verdict

Hard refutation. Forcing `--xla_tpu_enable_latency_hiding_layer_scheduler=true`
destroys the working memory schedule on the BKV2048 frontier and cannot compile.
Do not carry this setting.
