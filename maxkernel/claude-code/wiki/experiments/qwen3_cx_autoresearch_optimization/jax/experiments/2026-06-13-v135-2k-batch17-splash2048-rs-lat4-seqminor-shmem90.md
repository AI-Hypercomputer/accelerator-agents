---
title: "Qwen3 JAX current v135 - 2k batch17 all SEQ_MINOR shmem90"
type: experiment
hypothesis: "The all-SEQ_MINOR layout may make the previously fitted batch17/shmem90 branch throughput-competitive."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v130-splash-layout-env
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, layout, data-formatting, sparsecore, collectives, async-collective-fusion, reduce-scatter, latency-model, scoped-vmem, shmem90, batch17, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v135 - 2k batch17 all SEQ_MINOR shmem90

## Hypothesis under test

**Hypothesis**: v130's all-Q/K/V `SEQ_MINOR` layout reduced HLO copy tokens and
raised batch16 throughput by about 3.6%. The old batch17/shmem90 branch fit but
was too slow on the pre-layout graph. Combining batch17/shmem90 with all
`SEQ_MINOR` may amortize work enough to beat v130 while staying inside runtime
memory.

**Mechanism**: Start from v130 and change only:

```bash
--batch_size=17
--xla_tpu_scheduler_percent_shared_memory_limit=90
```

Keep all-Q/K/V `SEQ_MINOR`, Splash BQ/BKV/BKV_COMPUTE 2048, RS latency
multiplier 4, scheduler rerun 4, VMEM 98304, and the frontier ACF/SparseCore
settings fixed.

## Setup

- Workload: `alekseyv-qwen3-v135-2kbs17seqsh90`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v130-splash-layout-env`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v135-2k-bs17-splash2048-rslat4-seqminor-sh90`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v135_2k_bs17_splash2048_rslat4_seqminor_sh90`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.
- Mesh: `fsdp=8,tp=1`; per-chip batch 17 / global batch 136; sequence length
  2048.

## Results

- Worker 0: 73,579 tok/s global, 9,197 tok/s/chip, approx 47.3% MFU.
- Worker 1: 73,577 tok/s global, 9,197 tok/s/chip, approx 47.3% MFU.
- HLO SHA256:
  `bede1203771f2043f4fd05ce4850c14dedd2e6c9fa5e588866a10c19c8ebbd8a`.
- HLO size: 19,373,879 bytes / 115,102 lines.
- HLO op-token counts: all-gather=7060, all-reduce=165,
  reduce-scatter=2774, async-start=373, async-done=373, convolution=3071,
  custom-call=3824, copy=16174, fusion=30243, dot_general=7284, tokamax=1.
- Compiled memory report: 28.77 GiB total, 23.01 GiB temp.

## Verdict

Refuted as a frontier improvement, but useful as a near tie. Batch17/shmem90
fits with all-Q/K/V `SEQ_MINOR` and recovers nearly all of v130's throughput,
but the best worker remains a few tokens/sec below v130. The low copy-token
count says the all-`SEQ_MINOR` layout carries correctly to batch17; the
remaining issue is schedule quality from the shmem90 cap.
