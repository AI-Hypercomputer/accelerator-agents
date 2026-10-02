---
title: "Qwen3 JAX current v106 - 2k batch16 Splash BKV2048 no reduce-scatter ACF"
type: experiment
hypothesis: "Disabling reduce-scatter async-collective fusion on the confirmed BKV2048 frontier may reduce scheduler pressure or HBM tightness while preserving the all-gather ACF path that prior experiments showed is required."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, scheduler, async-collective-fusion, reduce-scatter, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v106 - 2k batch16 Splash BKV2048 no reduce-scatter ACF

## Hypothesis under test

**Hypothesis**: Prior scheduler sweeps showed that base async-collective fusion,
all-gather ACF, and multi-step ACF are required or at least not safely
removable. Reduce-scatter ACF has been less clearly useful on nearby frontiers.
On the current BKV2048 stack, disabling only reduce-scatter ACF may reduce
scheduler pressure, copies, or HBM pressure without disturbing the all-gather
path.

**Mechanism**: Reuse v095/v096 exactly and change only:

```bash
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
```

Keep `--batch_size=16`, `--seqlen=2048`, scoped VMEM **98304 KiB**,
`--xla_tpu_scheduler_percent_shared_memory_limit=100`, no-scan remat,
activation sharding, Tokamax Splash, MaxText CE, SparseCore reduce-scatter and
all-reduce offload, SparseCore collective aggregator, all-gather ACF enabled,
multiple-step ACF enabled, and:

```bash
export SPLASH_BKV=2048
export SPLASH_BKV_COMPUTE=2048
```

**Predicted signal**: Support requires beating the confirmed v095/v096
**70,959-70,976 tok/s** band or matching it with lower XProf step time or a
meaningfully lower custom-call/loop-fusion share. A same-HLO in-band runtime
refutes this axis.

## Setup

- Workload: `alekseyv-qwen3-jax-v106-2ksp2048noacfrs`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v106-2k-bs16-vmem98304-splash2048-noacfrs`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v106_2k_bs16_vmem98304_splash2048_noacfrs`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 16 / global batch 128; sequence length
  **2048**; profile window steps 12-14.

## Status

Completed on 2026-06-13.

## Results

- Workload: `alekseyv-qwen3-jax-v106-2ksp2048noacfrs`
- Worker throughput:
  - worker0: **70,945 tok/s**, **8,868 tok/s/chip**, **45.6%** script MFU.
  - worker1: **70,941 tok/s**, **8,868 tok/s/chip**, **45.6%** script MFU.
- XProf:
  `2026-06-13-qwen3-jax-v106-2k-bs16-vmem98304-splash2048-noacfrs/2026_06_13_05_41_03`
  - Step time: **3726.5 ms** average.
  - MXU utilization: **67.8%**.
  - Idle: **0.2%**.
- Op profile:
  - `convolution fusion`: **60,202.129 ms / 67.4%**.
  - `data formatting`: **3,541.861 ms / 4.0%** in the truncated top-op
    extract. The remaining custom-call and loop-fusion buckets stayed
    in-family with v095-v105.
- Memory profile: **31.14 GiB / 31.25 GiB** peak, **0.1018 GiB** free,
  **23.2933 GiB** stack, **7.8510 GiB** heap.
- HLO: `/tmp/qwen3-v106-hlo/`
  - SHA256:
    `a44dcef073d4f52793299c1a64717bbcf09fce4d2b8a3d73b21248f137cc6226`.
  - Size: **19,375,306 bytes / 115,424 lines**.
  - Compiled memory: **27.88 GiB** total, **22.12 GiB** preallocated temp.
  - Counts: `all-gather=7072`, `all-reduce=165`, `reduce-scatter=2774`,
    `async-start=391`, `async-done=391`, `convolution=3071`,
    `custom-call=3846`, `copy=17231`, `fusion=30274`,
    `dot_general=7301`, `tokamax=1`.
  - Diff versus v105 is instruction renumbering only in the sampled text diff;
    operation counts and memory are unchanged.

## Verdict

Refuted for speed. Disabling reduce-scatter ACF does not improve runtime,
profile step time, memory headroom, or the structural HLO. Keep
`--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true` on the
current BKV2048 frontier.
