---
title: "Qwen3 JAX current v136 - 2k batch17 all SEQ_MINOR shmem95"
type: experiment
hypothesis: "Batch17 all-SEQ_MINOR may beat v130 if shmem95 fits and improves the schedule over v135 shmem90."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v130-splash-layout-env
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, layout, data-formatting, sparsecore, collectives, async-collective-fusion, reduce-scatter, latency-model, scoped-vmem, shmem95, batch17, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v136 - 2k batch17 all SEQ_MINOR shmem95

## Hypothesis under test

**Hypothesis**: v135 proved batch17 all-`SEQ_MINOR` fits at shmem90 and almost
ties v130, but shmem90 may be over-constraining the scheduler. Raising the
shared-memory cap to 95 may recover enough schedule quality to beat v130 while
still avoiding the direct batch17/shmem100 memory failure observed before
v130.

**Mechanism**: Start from v135 and change only:

```bash
--xla_tpu_scheduler_percent_shared_memory_limit=95
```

Keep per-chip batch 17, all-Q/K/V `SEQ_MINOR`, Splash BQ/BKV/BKV_COMPUTE 2048,
RS latency multiplier 4, scheduler rerun 4, VMEM 98304, and the frontier
ACF/SparseCore settings fixed.

## Setup

- Workload: `alekseyv-qwen3-v136-2kbs17seqsh95`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v130-splash-layout-env`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v136-2k-bs17-splash2048-rslat4-seqminor-sh95`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v136_2k_bs17_splash2048_rslat4_seqminor_sh95`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.
- Mesh: `fsdp=8,tp=1`; per-chip batch 17 / global batch 136; sequence length
  2048.

## Results

- Worker 0: 73,587 tok/s global, 9,198 tok/s/chip, approx 47.3% MFU.
- Worker 1: 73,580 tok/s global, 9,198 tok/s/chip, approx 47.3% MFU.
- HLO SHA256:
  `b78961298603163ac732723bbbd718731d73864426456267aa9a0df5db37f8be`.
- HLO size: 19,381,301 bytes / 115,147 lines.
- HLO op-token counts: all-gather=7063, all-reduce=165,
  reduce-scatter=2774, async-start=373, async-done=373, convolution=3071,
  custom-call=3828, copy=16157, fusion=30258, dot_general=7288, tokamax=1.
- Compiled memory report: 29.08 GiB total, 23.33 GiB temp.

## Verdict

Provisionally supported. The gain over v130 is tiny, but v136 is the current
raw-throughput high-water mark and beats v135 while staying within runtime
memory. Because the margin is only 5 tok/s over v130, treat this as a frontier
candidate that needs adjacent shmem bracketing or rerun confirmation before
calling it durable.
