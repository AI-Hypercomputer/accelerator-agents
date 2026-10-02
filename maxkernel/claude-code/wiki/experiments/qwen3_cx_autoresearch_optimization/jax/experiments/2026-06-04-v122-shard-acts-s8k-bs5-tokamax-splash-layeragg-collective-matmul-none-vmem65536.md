---
title: "Qwen3 JAX v122 - shard acts s8k bs5 tokamax splash layeragg collective matmul none vmem65536"
type: experiment
hypothesis: "A lower scoped VMEM limit of 65536 KiB can make the batch-5 near-fit OOM compile."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, vmem, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v122 - shard acts s8k bs5 tokamax splash layeragg collective matmul none vmem65536

## Hypothesis under test

**Hypothesis**: v121's scoped-VMEM reduction helped the batch-5 OOM margin only
slightly but in the right direction. Lowering scoped VMEM further to 65536 KiB
may be enough to make batch 5/global 40 fit.

**Mechanism**: Keep v121 fixed and change only:

```text
--xla_tpu_scoped_vmem_limit_kib=73728 -> 65536
```

**Predicted signal**: Compile succeeds and the run reaches finite-loss steps. If
valid, compare total throughput to v118's 56,445 tok/s / 40.5% MFU frontier.

**Falsification criterion**: OOM/compile failure, invalid loss, or clean
completion that does not beat v118 throughput. If this still OOMs, batch 5
likely needs structural memory work rather than scalar VMEM/shared-memory knobs.

## Setup

GKE workload `alekseyv-qwen3-v122-bs5-vmem65536`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v122-shard-acts-s8k-bs5-tokamax-splash-layeragg-collective-matmul-none-vmem65536`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v121 stack, except scoped VMEM is 65536 KiB.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=5 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 40
- **Profile window**: steps 12-14

## Results

Failed during train-step compile before any measured step (`EXIT_CODE=1` on both
workers):

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 31.28G of 31.25G hbm.
Exceeded hbm capacity by 33.24M.
```

This is only about 4 MiB better than v121 and still does not fit.

## Profile

No runtime profile was captured because the workload failed at compile time.

## HLO Dump

Train-step OOM artifacts were emitted under:

```text
module_0267.jit_train_step.cl_854318611
```

- Optimized train-step text size: 805,927 bytes.
- HLO content hash: `d824280a1d9a97f4e70db3fa5c68fa01dcf501fb1154b2399fe22a8a9196e2f8`.
- HLO prefix object count: 1,289.
- Flagfile confirms both collective-matmul modes are `NONE`, scheduler
  shared-memory limit is 100, and scoped VMEM is 65536 KiB.

## Verdict

**Invalid, retire scalar batch-5 fit attempts.** Lowering scoped VMEM from 73728
to 65536 improves the OOM margin by only about 4 MiB and still misses capacity
by 33.24 MiB. Combined with v120's shmem90 regression, scalar scheduler/VMEM
knobs are not enough to make batch 5/global 40 valid. Keep the current frontier
at batch 4/global 32 and rerun v118 before banking the small gain.
