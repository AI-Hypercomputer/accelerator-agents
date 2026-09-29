---
title: "Qwen3 JAX current v134 - 2k batch16 Splash V-only SEQ_MINOR layout"
type: experiment
hypothesis: "V SEQ_MINOR may explain the remaining non-additive component of v130's all-layout win."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v130-splash-layout-env
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, layout, data-formatting, sparsecore, collectives, async-collective-fusion, reduce-scatter, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v134 - 2k batch16 Splash V-only SEQ_MINOR layout

## Hypothesis under test

**Hypothesis**: Q-only is better than K-only or K/V-only but still below v130.
V-only may expose whether the non-additive win requires V layout, or whether
the benefit only appears when Q/K/V are all converted together.

**Mechanism**: Reuse the v130 layout-env image and the v119 frontier settings,
but set only:

```bash
TOKAMAX_V_LAYOUT=SEQ_MINOR
```

Leave `TOKAMAX_Q_LAYOUT` and `TOKAMAX_K_LAYOUT` at default `HEAD_DIM_MINOR`.

## Setup

- Workload: `alekseyv-qwen3-v134-2kvseq`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v130-splash-layout-env`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v134-2k-bs16-splash2048-rslat4-vseq`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v134_2k_bs16_splash2048_rslat4_vseq`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker 0: 71,119 tok/s global, 8,890 tok/s/chip, approx 45.7% MFU.
- Worker 1: 71,129 tok/s global, 8,891 tok/s/chip, approx 45.7% MFU.
- HLO SHA256:
  `4bdade7017b817c11c17ca3cd95c66ae51d327e60b10ae1efe1cbfbf52b0d9b4`.
- HLO size: 19,342,214 bytes / 115,207 lines.
- HLO op-token counts: all-gather=7063, all-reduce=165,
  reduce-scatter=2774, async-start=379, async-done=379, convolution=3071,
  custom-call=3834, copy=17165, fusion=30238, dot_general=7288, tokamax=1.
- Compiled memory report: 27.88 GiB total, 22.12 GiB temp.

## Verdict

Refuted relative to v130. V-only `SEQ_MINOR` is slower than Q-only, K-only, and
K/V-only, and its copy-token count is still much higher than v130. The layout
sweep supports carrying all-Q/K/V `SEQ_MINOR`; no single operand or K/V subset
captures the gain.
