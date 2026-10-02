---
title: "Qwen3 JAX current v109 - 2k batch16 Splash BKV2048 async reduce-scatter cap1"
type: experiment
hypothesis: "Limiting concurrent async reduce-scatters to one may reduce scheduling pressure on the BKV2048 frontier's SparseCore reduce-scatter offload path."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, scheduler, async-collective-fusion, reduce-scatter, concurrency, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v109 - 2k batch16 Splash BKV2048 async reduce-scatter cap1

## Hypothesis under test

**Hypothesis**: The current BKV2048 frontier has `all-gather` concurrency
resolved to 1 already, while `--xla_max_concurrent_async_reduce_scatters`
remains `auto`. The HLO still contains **2,774** reduce-scatter references and
keeps reduce-scatter offloaded to SparseCore. Constraining async
reduce-scatter concurrency to 1 may reduce scheduler pressure or confirm that
the `auto` setting is already optimal.

**Mechanism**: Reuse v095/v096 exactly and add only:

```bash
--xla_max_concurrent_async_reduce_scatters=1
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

- Workload: `alekseyv-qwen3-jax-v109-2ksp2048rs1`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v109-2k-bs16-vmem98304-splash2048-rs1`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v109_2k_bs16_vmem98304_splash2048_rs1`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 16 / global batch 128; sequence length
  **2048**; profile window steps 12-14.

## Status

Completed on 2026-06-13.

## Results

- Workload `alekseyv-qwen3-jax-v109-2ksp2048rs1` completed cleanly with
  `EXIT_CODE=0`.
- Worker throughput:
  - worker0: **70,946 tok/s global**, **8,868 tok/s/chip**, **45.6% MFU**.
  - worker1: **70,963 tok/s global**, **8,870 tok/s/chip**, **45.6% MFU**.
- XProf run:
  `2026-06-13-qwen3-jax-v109-2k-bs16-vmem98304-splash2048-rs1/2026_06_13_06_27_55`.
- XProf summary: **3724.2 ms** step time, **67.8%** MXU utilization,
  **0.2%** idle.
- Runtime memory: **31.14 GiB / 31.25 GiB** peak use, **0.1018 GiB** free,
  **23.2933 GiB** stack reservation, **7.851 GiB** heap, **1.92%**
  fragmentation.
- Op profile remains in-family: convolution fusion **60,173.749 ms / 67.4%**,
  custom-call **11,695.276 ms / 13.1%**, loop fusion **10,923.254 ms /
  12.2%**, forward residuals **5,905.249 ms**, and DKV total **5,788.346 ms**.
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

Refuted / inert. The runtime is inside the v095/v096 noise band and the HLO is
byte-identical to the confirmed frontier. Do not carry the explicit
`--xla_max_concurrent_async_reduce_scatters=1`; the current `auto` setting is
already equivalent for this frontier.
