---
title: "Qwen3 JAX current v132 - 2k batch16 Splash K/V-only SEQ_MINOR layout"
type: experiment
hypothesis: "K/V SEQ_MINOR may account for the incremental gain missing from v131's Q-only layout."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v130-splash-layout-env
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, layout, data-formatting, sparsecore, collectives, async-collective-fusion, reduce-scatter, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v132 - 2k batch16 Splash K/V-only SEQ_MINOR layout

## Hypothesis under test

**Hypothesis**: v131 showed Q-only `SEQ_MINOR` is not enough to match v130.
The remaining v130 gain may come from K/V layout, despite the smaller K/V head
count, because the Splash backward path is sensitive to K/V physical layout.

**Mechanism**: Reuse the v130 layout-env image and the v119 frontier settings,
but set only:

```bash
TOKAMAX_K_LAYOUT=SEQ_MINOR
TOKAMAX_V_LAYOUT=SEQ_MINOR
```

Leave `TOKAMAX_Q_LAYOUT` at default `HEAD_DIM_MINOR`.

## Setup

- Workload: `alekseyv-qwen3-v132-2kkvseq`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v130-splash-layout-env`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v132-2k-bs16-splash2048-rslat4-kvseq`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v132_2k_bs16_splash2048_rslat4_kvseq`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker 0: 71,601 tok/s global, 8,950 tok/s/chip, approx 46.0% MFU.
- Worker 1: 71,601 tok/s global, 8,950 tok/s/chip, approx 46.0% MFU.
- HLO SHA256:
  `3028daac1b565bb7860390679ec90ff208b2ba9670bb7f3a32a57166b97c2338`.
- HLO size: 19,373,572 bytes / 115,172 lines.
- HLO op-token counts: all-gather=7063, all-reduce=165,
  reduce-scatter=2774, async-start=375, async-done=375, convolution=3071,
  custom-call=3831, copy=17107, fusion=30259, dot_general=7288, tokamax=1.
- Compiled memory report: 27.88 GiB total, 22.12 GiB temp.

## Verdict

Refuted relative to v130. K/V-only `SEQ_MINOR` does not reproduce the all-layout
win and regresses below v131 Q-only. The copy-token count remains much closer to
the old frontier than to v130, so the v130 win appears to require Q plus K/V
layout changes together rather than a K/V-only formatting improvement.
