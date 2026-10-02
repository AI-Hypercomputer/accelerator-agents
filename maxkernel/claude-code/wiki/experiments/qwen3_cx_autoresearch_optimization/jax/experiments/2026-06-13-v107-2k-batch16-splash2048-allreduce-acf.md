---
title: "Qwen3 JAX current v107 - 2k batch16 Splash BKV2048 all-reduce ACF"
type: experiment
hypothesis: "Enabling all-reduce async-collective fusion on the confirmed BKV2048 frontier may hide residual optimizer/reduction collectives without disturbing the required all-gather and reduce-scatter ACF paths."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, scheduler, async-collective-fusion, all-reduce, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v107 - 2k batch16 Splash BKV2048 all-reduce ACF

## Hypothesis under test

**Hypothesis**: v106 showed reduce-scatter ACF should remain enabled. The
frontier still leaves a small all-reduce population in the HLO and uses
SparseCore all-reduce offload. Enabling all-reduce ACF may improve residual
collective hiding or be absorbed as a harmless no-op; it is a narrow scheduler
probe that preserves the confirmed all-gather and reduce-scatter ACF settings.

**Mechanism**: Reuse v095/v096 exactly and change only:

```bash
--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true
```

Keep `--batch_size=16`, `--seqlen=2048`, scoped VMEM **98304 KiB**,
`--xla_tpu_scheduler_percent_shared_memory_limit=100`, no-scan remat,
activation sharding, Tokamax Splash, MaxText CE, SparseCore reduce-scatter and
all-reduce offload, SparseCore collective aggregator, all-gather ACF enabled,
reduce-scatter ACF enabled, multiple-step ACF enabled, and:

```bash
export SPLASH_BKV=2048
export SPLASH_BKV_COMPUTE=2048
```

**Predicted signal**: Support requires beating the confirmed v095/v096
**70,959-70,976 tok/s** band or matching it with lower XProf step time.
Same-HLO in-band runtime means all-reduce ACF is inert on this frontier.

## Setup

- Workload: `alekseyv-qwen3-jax-v107-2ksp2048acf-ar`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v107-2k-bs16-vmem98304-splash2048-acf-ar`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v107_2k_bs16_vmem98304_splash2048_acf_ar`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 16 / global batch 128; sequence length
  **2048**; profile window steps 12-14.

## Status

Completed on 2026-06-13.

## Results

- Workload: `alekseyv-qwen3-jax-v107-2ksp2048acf-ar`
- Worker throughput:
  - worker0: **70,943 tok/s**, **8,868 tok/s/chip**, **45.6%** script MFU.
  - worker1: **70,945 tok/s**, **8,868 tok/s/chip**, **45.6%** script MFU.
- XProf:
  `2026-06-13-qwen3-jax-v107-2k-bs16-vmem98304-splash2048-acf-ar/2026_06_13_05_56_10`
  - Step time: **3725.1 ms** average.
  - MXU utilization: **67.8%**.
  - Idle: **0.2%**.
- Op profile:
  - `convolution fusion`: **60,190.155 ms / 67.4%**.
  - `custom-call`: **11,695.236 ms / 13.1%**.
  - `loop fusion`: **10,921.098 ms / 12.2%**.
  - Forward residuals: **5,905.228 ms**.
  - DKV total: **5,788.328 ms**.
- Memory profile: **31.14 GiB / 31.25 GiB** peak, **0.1018 GiB** free,
  **23.2933 GiB** stack, **7.8510 GiB** heap.
- HLO: `/tmp/qwen3-v107-hlo/`
  - SHA256:
    `a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`.
  - Size: **19,375,297 bytes / 115,424 lines**.
  - Compiled memory: **27.88 GiB** total, **22.12 GiB** preallocated temp.
  - Counts: `all-gather=7072`, `all-reduce=165`, `reduce-scatter=2774`,
    `async-start=391`, `async-done=391`, `convolution=3071`,
    `custom-call=3846`, `copy=17231`, `fusion=30274`,
    `dot_general=7301`, `tokamax=1`.

## Verdict

Refuted / inert. Enabling all-reduce ACF does not beat the confirmed v095/v096
frontier and does not change the optimized train-step HLO. Keep
`--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false` on the current
BKV2048 frontier.
