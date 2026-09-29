---
title: "Qwen3 JAX current v131 - 2k batch16 Splash Q-only SEQ_MINOR layout"
type: experiment
hypothesis: "Q layout may account for most of the v130 SEQ_MINOR win because the largest attention formatting copies are query-head shaped."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v130-splash-layout-env
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, layout, data-formatting, sparsecore, collectives, async-collective-fusion, reduce-scatter, latency-model, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v131 - 2k batch16 Splash Q-only SEQ_MINOR layout

## Hypothesis under test

**Hypothesis**: v130's all-operand `SEQ_MINOR` layout is supported and faster,
but the dominant copy signatures are query-head shaped. Setting only the query
layout to `SEQ_MINOR` may preserve most of the throughput gain while avoiding
unnecessary K/V layout changes.

**Mechanism**: Reuse the v130 layout-env image and the v119 frontier settings,
but set only:

```bash
TOKAMAX_Q_LAYOUT=SEQ_MINOR
```

Leave `TOKAMAX_K_LAYOUT` and `TOKAMAX_V_LAYOUT` at their default
`HEAD_DIM_MINOR`.

## Setup

- Workload: `alekseyv-qwen3-v131-2kqseq`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v130-splash-layout-env`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v131-2k-bs16-splash2048-rslat4-qseq`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v131_2k_bs16_splash2048_rslat4_qseq`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker 0: 72,883 tok/s global, 9,110 tok/s/chip, approx 46.9% MFU.
- Worker 1: 72,890 tok/s global, 9,111 tok/s/chip, approx 46.9% MFU.
- HLO SHA256:
  `8445702d8654166be7694a400e695e916eaabc75774b189b0d81eef82514c570`.
- HLO size: 19,421,298 bytes / 115,453 lines.
- HLO op-token counts: all-gather=7063, all-reduce=165,
  reduce-scatter=2774, async-start=373, async-done=373, convolution=3071,
  custom-call=3828, copy=17861, fusion=30239, dot_general=7288, tokamax=1.
- Compiled memory report: 27.88 GiB total, 22.12 GiB temp.

## Verdict

Refuted relative to v130. Q-only `SEQ_MINOR` is faster than the old v119
frontier but loses about 0.4 MFU points versus all-operand `SEQ_MINOR`, and its
copy-token count is worse than the old frontier. Keep v130's all-Q/K/V
`SEQ_MINOR` setting.
