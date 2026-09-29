---
title: "Qwen3 JAX current v108 - 2k batch16 Splash BKV2048 async all-gather cap1"
type: experiment
hypothesis: "Limiting concurrent async all-gathers to one on the tight-HBM BKV2048 frontier may reduce residual all-gather scheduling pressure without changing model math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, scheduler, async-collective-fusion, all-gather, concurrency, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v108 - 2k batch16 Splash BKV2048 async all-gather cap1

## Hypothesis under test

**Hypothesis**: v106/v107 showed the current BKV2048 frontier is insensitive to
reduce-scatter and all-reduce ACF sub-flag flips. The remaining visible
collective pressure is all-gather-heavy fused matmul work. Historical cap1
runs were mostly inert/tie, while cap2 regressed by exposing direct
all-gather. On the tight-HBM 2k BKV2048 frontier, cap1 may still improve
runtime scheduling slightly without changing HLO structure.

**Mechanism**: Reuse v095/v096 exactly and add only:

```bash
--xla_max_concurrent_async_all_gathers=1
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
**70,959-70,976 tok/s** band or matching it with lower XProf step time or lower
all-gather/convolution-fusion tail. Same-HLO in-band runtime refutes the cap.

## Setup

- Workload: `alekseyv-qwen3-jax-v108-2ksp2048ag1`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v108-2k-bs16-vmem98304-splash2048-ag1`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v108_2k_bs16_vmem98304_splash2048_ag1`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 16 / global batch 128; sequence length
  **2048**; profile window steps 12-14.

## Status

Completed on 2026-06-13.

## Results

- Workload: `alekseyv-qwen3-jax-v108-2ksp2048ag1`
- Worker throughput:
  - worker0: **70,935 tok/s**, **8,867 tok/s/chip**, **45.6%** script MFU.
  - worker1: **70,939 tok/s**, **8,867 tok/s/chip**, **45.6%** script MFU.
- XProf:
  `2026-06-13-qwen3-jax-v108-2k-bs16-vmem98304-splash2048-ag1/2026_06_13_06_11_14`
  - Step time: **3726.6 ms** average.
  - MXU utilization: **67.8%**.
  - Idle: **0.2%**.
- Op profile:
  - `convolution fusion`: **60,194.825 ms / 67.4%**.
  - `custom-call`: **11,695.335 ms / 13.1%**.
  - `loop fusion`: **10,922.114 ms / 12.2%**.
  - Forward residuals: **5,905.324 ms**.
  - DKV total: **5,788.331 ms**.
- Memory profile: **31.14 GiB / 31.25 GiB** peak, **0.1018 GiB** free,
  **23.2933 GiB** stack, **7.8510 GiB** heap.
- HLO: `/tmp/qwen3-v108-hlo/`
  - SHA256:
    `a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`.
  - Size: **19,375,297 bytes / 115,424 lines**.
  - Compiled memory: **27.88 GiB** total, **22.12 GiB** preallocated temp.
  - Counts: `all-gather=7072`, `all-reduce=165`, `reduce-scatter=2774`,
    `async-start=391`, `async-done=391`, `convolution=3071`,
    `custom-call=3846`, `copy=17231`, `fusion=30274`,
    `dot_general=7301`, `tokamax=1`.
- Flagfile check: v095 already resolved
  `--xla_max_concurrent_async_all_gathers=1`, so this run made the default
  explicit rather than changing the compiled/runtime setting.

## Verdict

Refuted / explicit-default rerun. The async all-gather cap was already **1** in
the confirmed v095 frontier flagfile, so explicitly setting it does not change
HLO, memory, or profile shape and does not beat the frontier. No carry change.
