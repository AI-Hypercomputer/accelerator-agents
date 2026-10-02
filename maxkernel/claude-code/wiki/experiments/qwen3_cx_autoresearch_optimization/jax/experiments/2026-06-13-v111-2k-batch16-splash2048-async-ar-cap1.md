---
title: "Qwen3 JAX current v111 - 2k batch16 Splash BKV2048 async all-reduce cap1"
type: experiment
hypothesis: "Limiting concurrent async all-reduces to one may reduce scheduler pressure on the BKV2048 frontier without changing the successful layer schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, scheduler, async-collective-fusion, all-reduce, concurrency, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v111 - 2k batch16 Splash BKV2048 async all-reduce cap1

## Hypothesis under test

**Hypothesis**: The current BKV2048 frontier leaves
`--xla_max_concurrent_async_all_reduces=auto`. The train-step HLO still
contains **165** all-reduce references, mostly from the SparseCore all-reduce
offload path. Constraining concurrent async all-reduces to one may reduce
scheduler pressure without forcing the layer scheduler, which v110 showed is
unsafe.

**Mechanism**: Reuse v095/v096 exactly and add only:

```bash
--xla_max_concurrent_async_all_reduces=1
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
same-HLO in-band runtime refutes the cap.

## Setup

- Workload: `alekseyv-qwen3-jax-v111-2ksp2048ar1`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v111-2k-bs16-vmem98304-splash2048-ar1`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v111_2k_bs16_vmem98304_splash2048_ar1`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 16 / global batch 128; sequence length
  **2048**; profile window steps 12-14.

## Status

Completed on 2026-06-13.

## Results

- Workload `alekseyv-qwen3-jax-v111-2ksp2048ar1` completed cleanly with
  `EXIT_CODE=0`.
- Worker throughput:
  - worker0: **70,935 tok/s global**, **8,867 tok/s/chip**, **45.6% MFU**.
  - worker1: **70,941 tok/s global**, **8,868 tok/s/chip**, **45.6% MFU**.
- Best early steady steps reached about **71,540 tok/s**, but the profile
  window and final average drifted below the confirmed v095/v096 frontier.
- XProf run queried with xprof pointed directly at the v111 run directory:
  `2026_06_13_06_56_24`.
- XProf summary: **3724.4 ms** step time, **67.8%** MXU utilization,
  **0.2%** idle in the op profile.
- Runtime memory: **31.14 GiB / 31.25 GiB** peak use, **0.1018 GiB** free,
  **23.2933 GiB** stack reservation, **7.851 GiB** heap, **1.92%**
  fragmentation.
- Op profile remains in-family: convolution fusion **60,185.150 ms / 67.4%**,
  total program time **89,094.721 ms**, idle **188.882 ms / 0.2%**.
- Optimized train-step HLO:
  - SHA256:
    `a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`.
  - Size: **19,375,297 bytes / 115,424 lines**.
  - Counts: `all-gather=7072`, `all-reduce=165`,
    `reduce-scatter=2774`, `async-start=391`, `async-done=391`,
    `convolution=3071`, `custom-call=3846`, `copy=17231`,
    `fusion=30274`, `dot_general=7301`, `tokamax=1`.
  - Compiled memory report: **27.88 GiB** total, **22.12 GiB**
    preallocated temp.

## Verdict

Refuted / inert. The all-reduce concurrency cap does not change the optimized
train-step HLO and does not improve the profile step time. Do not carry
`--xla_max_concurrent_async_all_reduces=1`.
