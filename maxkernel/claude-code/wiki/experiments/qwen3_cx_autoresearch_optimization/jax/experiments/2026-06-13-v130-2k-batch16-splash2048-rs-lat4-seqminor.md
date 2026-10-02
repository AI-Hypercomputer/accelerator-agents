---
title: "Qwen3 JAX current v130 - 2k batch16 Splash SEQ_MINOR layout probe"
type: experiment
hypothesis: "Using Tokamax Splash SEQ_MINOR Q/K/V layouts may reduce attention data-formatting copies visible in the v129/v119 profile."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v130-splash-layout-env
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, layout, data-formatting, sparsecore, collectives, async-collective-fusion, reduce-scatter, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v130 - 2k batch16 Splash SEQ_MINOR layout probe

## Hypothesis under test

**Hypothesis**: XProf shows a persistent data-formatting bucket around attention
backward/remat copies and reshapes. Tokamax Splash supports `SEQ_MINOR` as an
alternate Q/K/V layout; selecting it for all three operands may align better
with the model's sequence-major projected tensors and reduce copy time.

**Mechanism**: Derive a thin image from v067 that changes only
`splash_attn.py` to read `TOKAMAX_Q_LAYOUT`, `TOKAMAX_K_LAYOUT`, and
`TOKAMAX_V_LAYOUT` from the environment. Run the v119 frontier with:

```bash
TOKAMAX_Q_LAYOUT=SEQ_MINOR
TOKAMAX_K_LAYOUT=SEQ_MINOR
TOKAMAX_V_LAYOUT=SEQ_MINOR
```

Keep batch 16, sequence 2048, Splash BKV/BKV_COMPUTE 2048, VMEM 98304,
shmem100, scheduler rerun 4, async collective fusion enabled, all-gather and
reduce-scatter ACF enabled, all-reduce ACF disabled, RS latency multiplier 4,
AR latency multiplier 2, SparseCore all-gather offload disabled,
reduce-scatter/all-reduce offload enabled, and aggregator enabled.

## Setup

- Workload: `alekseyv-qwen3-v130-2kseqminor`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v130-splash-layout-env`
- Base image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v130-2k-bs16-splash2048-rslat4-seqminor`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v130_2k_bs16_splash2048_rslat4_seqminor`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker 0: 73,576 tok/s global, 9,197 tok/s/chip, approx 47.3% MFU.
- Worker 1: 73,582 tok/s global, 9,198 tok/s/chip, approx 47.3% MFU.
- Profile files:
  `plugins/profile/2026_06_13_11_31_10/`.
- XProf MCP query was pending/timing out at filing time; profile files are
  present in GCS for later analysis.
- HLO SHA256:
  `66462ca89d8bad5075122542dab49ce88a47ad9923e87107e81746cc4d23d831`.
- HLO size: 19,377,734 bytes / 115,099 lines.
- HLO op-token counts: all-gather=7063, all-reduce=165,
  reduce-scatter=2774, async-start=375, async-done=375, convolution=3071,
  custom-call=3831, copy=15956, fusion=30259, dot_general=7288, tokamax=1.
- Compiled memory report: 27.88 GiB total, 22.12 GiB temp.

## Verdict

Supported. Switching Tokamax Splash Q/K/V layouts to `SEQ_MINOR` is the new 2k
frontier. It reduces HLO copy-token count by 1,329 versus v119/v128 while
preserving the same high-level collective and convolution structure, lifting
throughput from v119's 71,053 tok/s to 73,582 tok/s.
