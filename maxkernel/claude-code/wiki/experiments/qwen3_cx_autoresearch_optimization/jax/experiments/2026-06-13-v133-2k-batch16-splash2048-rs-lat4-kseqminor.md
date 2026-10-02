---
title: "Qwen3 JAX current v133 - 2k batch16 Splash K-only SEQ_MINOR layout"
type: experiment
hypothesis: "K SEQ_MINOR may be the interaction trigger behind v130's all-layout win."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v130-splash-layout-env
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, layout, data-formatting, sparsecore, collectives, async-collective-fusion, reduce-scatter, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v133 - 2k batch16 Splash K-only SEQ_MINOR layout

## Hypothesis under test

**Hypothesis**: v131 Q-only is close to v130 but misses about 0.4 MFU points,
while v132 K/V-only regresses. The remaining interaction may be specifically
the K operand's layout in the QK path rather than V layout.

**Mechanism**: Reuse the v130 layout-env image and the v119 frontier settings,
but set only:

```bash
TOKAMAX_K_LAYOUT=SEQ_MINOR
```

Leave `TOKAMAX_Q_LAYOUT` and `TOKAMAX_V_LAYOUT` at default `HEAD_DIM_MINOR`.

## Setup

- Workload: `alekseyv-qwen3-v133-2kkseq`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v130-splash-layout-env`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v133-2k-bs16-splash2048-rslat4-kseq`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v133_2k_bs16_splash2048_rslat4_kseq`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker 0: 71,531 tok/s global, 8,941 tok/s/chip, approx 46.0% MFU.
- Worker 1: 71,529 tok/s global, 8,941 tok/s/chip, approx 46.0% MFU.
- HLO SHA256:
  `ac1cd1d2831f6b34fd97710d35a3a87a275586960475b5d7d638c59271560757`.
- HLO size: 19,380,786 bytes / 115,244 lines.
- HLO op-token counts: all-gather=7063, all-reduce=165,
  reduce-scatter=2774, async-start=375, async-done=375, convolution=3071,
  custom-call=3831, copy=17215, fusion=30259, dot_general=7288, tokamax=1.
- Compiled memory report: 27.88 GiB total, 22.12 GiB temp.

## Verdict

Refuted relative to v130. K-only `SEQ_MINOR` is close to v132 K/V-only and
well below v131 Q-only, while its copy-token count remains far above v130. K
layout alone is not the mechanism behind the all-operand `SEQ_MINOR` win.
