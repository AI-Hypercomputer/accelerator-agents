---
title: "Qwen3 JAX current v137 - 2k batch17 all SEQ_MINOR shmem97"
type: experiment
hypothesis: "Batch17 all-SEQ_MINOR may improve beyond v136 if shmem97 fits without crossing the runtime memory edge."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v130-splash-layout-env
status: running
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, layout, data-formatting, sparsecore, collectives, async-collective-fusion, reduce-scatter, latency-model, scoped-vmem, shmem97, batch17, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v137 - 2k batch17 all SEQ_MINOR shmem97

## Hypothesis under test

**Hypothesis**: v136 shmem95 fits and narrowly beats v130, but the margin is
tiny. Raising the shared-memory cap to 97 may improve schedule quality further
while staying below the old batch17/shmem100 runtime memory boundary.

**Mechanism**: Start from v136 and change only:

```bash
--xla_tpu_scheduler_percent_shared_memory_limit=97
```

Keep per-chip batch 17, all-Q/K/V `SEQ_MINOR`, Splash BQ/BKV/BKV_COMPUTE 2048,
RS latency multiplier 4, scheduler rerun 4, VMEM 98304, and the frontier
ACF/SparseCore settings fixed.

## Setup

- Workload: `alekseyv-qwen3-v137-2kbs17seqsh97`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v130-splash-layout-env`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v137-2k-bs17-splash2048-rslat4-seqminor-sh97`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v137_2k_bs17_splash2048_rslat4_seqminor_sh97`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.
- Mesh: `fsdp=8,tp=1`; per-chip batch 17 / global batch 136; sequence length
  2048.

## Results

Running.

## Verdict

Pending.
