---
title: "Qwen3 JAX current v096 - 2k batch16 shmem100 Splash BKV2048 rerun"
type: experiment
hypothesis: "v095 set a new 2k high-water mark but left only 0.10 GiB free HBM, so an exact rerun should determine whether BKV2048 is durable or a fragile noise sample."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, scheduler, scoped-vmem, shmem100, batch16, rerun, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v096 - 2k batch16 shmem100 Splash BKV2048 rerun

## Hypothesis under test

**Hypothesis**: v095's `SPLASH_BKV=2048 SPLASH_BKV_COMPUTE=2048` result is a
real 2k short-context attention-tile gain rather than a one-off runtime sample.

**Mechanism**: Exact rerun of v095, changing only workload identity, compile
cache, and profile/HLO output path. Keep `--batch_size=16`, `--seqlen=2048`,
scoped VMEM **98304 KiB**, shmem100, no-scan remat, activation sharding,
Tokamax Splash, MaxText CE, SparseCore reduce-scatter/all-reduce offload,
SparseCore collective aggregator, and the forward Splash tile overrides:

```bash
export SPLASH_BKV=2048
export SPLASH_BKV_COMPUTE=2048
```

**Predicted signal**: Support requires clean finite loss and another summary
near the v095 band (**70.97k tok/s**) with a matching HLO hash/profile shape.
OOM, runtime allocation failure, or a return to the v092/v093 **70.60k tok/s**
band refutes carrying BKV2048 without more memory work.

## Setup

- Workload: `alekseyv-qwen3-jax-v096-2ksp2048r`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v096-2k-bs16-vmem98304-shmem100-splash2048-rerun`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v096_2k_bs16_vmem98304_shmem100_splash2048_rerun`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 16 / global batch 128; sequence length
  **2048**; profile window steps 12-14.

## Status

Completed on 2026-06-13 with `EXIT_CODE=0` on both workers.

## Results

Worker1 reported **70,959 tok/s**, **8,870 tok/s/chip**, and **45.6% script
MFU**. Worker0 reported **70,957 tok/s**, **8,870 tok/s/chip**, and **45.6%
script MFU**. Loss stayed finite, ending at **12.0467** on both workers.

The log confirmed the intended Tokamax Splash config:

```text
bq=2048 bkv=2048 bq_dkv=2048 bkv_dkv=2048 fused_bwd=True
```

XProf run
`2026-06-13-qwen3-jax-v096-2k-bs16-vmem98304-shmem100-splash2048-rerun/2026_06_13_03_10_13`
shows **3723.7 ms** average step time and **67.8% MXU**. Top HLO buckets by
time were convolution fusion **60,179.211 ms / 67.4%**, custom-call
**11,696.061 ms / 13.1%**, loop fusion **10,928.655 ms / 12.2%**, data
formatting **3,543.761 ms / 4.0%**, and custom fusion **1,830.332 ms / 2.1%**.

Memory remains extremely tight: XProf memory profile reports peak HBM
**31.14 GiB / 31.25 GiB**, leaving only **0.1018 GiB** free. The optimized
train-step HLO is byte-identical to v095: **19,375,297 bytes / 115,424 lines**,
SHA256 `a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`,
with HLO memory total **27.88 GiB** and preallocated temp **22.12 GiB**.

## Verdict

**Supported / confirmed 2k frontier.** v096 confirms that v095's BKV2048 gain
is durable rather than a one-off sample: both workers stayed in the
**70.95k tok/s** band, the optimized HLO hash matches v095 exactly, and the
profile shape is stable. Carry
`SPLASH_BKV=2048 SPLASH_BKV_COMPUTE=2048` as the current 2k frontier.

The next hypothesis should address robustness around the same frontier rather
than assume there is much HBM slack: runtime free HBM is only **0.1018 GiB**.
