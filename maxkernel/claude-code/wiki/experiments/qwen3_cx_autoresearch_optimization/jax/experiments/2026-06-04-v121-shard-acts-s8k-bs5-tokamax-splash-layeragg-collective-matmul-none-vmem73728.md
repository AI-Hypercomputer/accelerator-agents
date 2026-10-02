---
title: "Qwen3 JAX v121 - shard acts s8k bs5 tokamax splash layeragg collective matmul none vmem73728"
type: experiment
hypothesis: "Lowering scoped VMEM can make the v119 batch-5 near-fit OOM compile without the shmem90 regression."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, vmem, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v121 - shard acts s8k bs5 tokamax splash layeragg collective matmul none vmem73728

## Hypothesis under test

**Hypothesis**: v119's batch-5/global-40 OOM missed by only 46.98 MiB. Lowering
the scoped VMEM limit from 81920 to 73728 KiB may choose a slightly smaller
schedule and make batch 5 fit, without the large-HBM regression seen in v120
with `scheduler_percent_shared_memory_limit=90`.

**Mechanism**: Keep v119 fixed and change only:

```text
--xla_tpu_scoped_vmem_limit_kib=81920 -> 73728
```

Restore `--xla_tpu_scheduler_percent_shared_memory_limit=100`.

**Predicted signal**: Compile succeeds and the run reaches finite-loss steps. If
valid, compare total throughput to v118's 56,445 tok/s / 40.5% MFU frontier.

**Falsification criterion**: OOM/compile failure, invalid loss, or clean
completion that does not beat v118 throughput.

## Setup

GKE workload `alekseyv-qwen3-v121-bs5-vmem73728`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v121-shard-acts-s8k-bs5-tokamax-splash-layeragg-collective-matmul-none-vmem73728`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v119 stack, except scoped VMEM is 73728 KiB.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=5 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 40
- **Profile window**: steps 12-14

## Results

Failed during train-step compile before any measured step (`EXIT_CODE=1` on both
workers):

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 31.28G of 31.25G hbm.
Exceeded hbm capacity by 37.55M.
```

This is slightly better than v119's 46.98 MiB OOM miss, but still does not fit.

## Profile

No runtime profile was captured because the workload failed at compile time.

## HLO Dump

Train-step OOM artifacts were emitted under:

```text
module_0267.jit_train_step.cl_854318611
```

- Optimized train-step text size: 790,117 bytes.
- HLO content hash: `c3dcd9083be4b8a7db2664908e9bb95779f01b2d348dc70786c6d3e24309a79e`.
- HLO prefix object count: 1,290.
- Flagfile confirms both collective-matmul modes are `NONE`, scheduler
  shared-memory limit is restored to 100, and scoped VMEM is 73728 KiB.

## Verdict

**Invalid, near-fit OOM.** Lowering scoped VMEM from 81920 to 73728 helps only
about 9 MiB versus v119 and still misses capacity by 37.55 MiB. The next lower
VMEM point, 65536 KiB, is worth one more direct fit test because the remaining
miss is still small; if that does not fit, batch 5 should be retired or require
a structural memory change rather than scalar scheduler knobs.
