---
title: "Qwen3 JAX v119 - shard acts s8k bs5 tokamax splash layeragg collective matmul none"
type: experiment
hypothesis: "Batch 5/global 40 may still fit with the collective-matmul-none stack and improve throughput beyond v118."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v119 - shard acts s8k bs5 tokamax splash layeragg collective matmul none

## Hypothesis under test

**Hypothesis**: The v118 batch-4 frontier leaves enough practical HBM margin for
per-chip batch 5 / global batch 40, or at least close enough that a direct fit
test is more informative than extrapolation.

**Mechanism**: Keep the v118 stack fixed and change only:

```text
--batch_size=4 -> --batch_size=5
```

**Predicted signal**: If it fits, total throughput should exceed v118's
56,445 tok/s / 40.5% MFU frontier. A compile-time or startup OOM is also useful
because v118 leaves only 4.34 GiB free and linear stack growth predicts a
near-capacity or slightly over-capacity point.

**Falsification criterion**: OOM/compile failure, invalid loss, or clean
completion that does not beat v118 total throughput.

## Setup

GKE workload `alekseyv-qwen3-v119-bs5-cmnone`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v119-shard-acts-s8k-bs5-tokamax-splash-layeragg-collective-matmul-none`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed collective-matmul-none stack from v116-v118.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=5 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 40
- **Profile window**: steps 12-14

## Results

Failed during train-step compile before any measured step (`EXIT_CODE=1` on both
workers):

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 31.29G of 31.25G hbm.
Exceeded hbm capacity by 46.98M.
```

The failure is close to the device limit, which makes this a useful fit signal
rather than a broad shape rejection.

## Profile

No runtime profile was captured because the workload failed at compile time.

## HLO Dump

Train-step OOM artifacts were emitted:

```text
module_0112.jit_train_step.cl_854318611.after_optimizations.txt
module_0112.jit_train_step.cl_854318611.oom_intermediate_module.txt
module_0112.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions
```

- Optimized train-step text size: 790,290 bytes.
- HLO content hash: `67b888766a137eab085ae4c419a9dc7369e263b7ff770b53ef891bae618b1c0d`.
- HLO prefix object count: 50.
- Flagfile confirms both collective-matmul modes are `NONE`, scoped VMEM is
  81920 KiB, layer scheduler and sparse-core collective aggregator are enabled,
  and scheduler shared-memory limit is still 100.

## Verdict

**Invalid, near-fit OOM.** Batch 5/global 40 does not fit under the v118 stack,
but it misses by only 46.98 MiB. The next most targeted probe is to keep batch 5
fixed and lower only `--xla_tpu_scheduler_percent_shared_memory_limit` from 100
to 90. Earlier bs3 testing showed this can choose a lower-HBM schedule, albeit
with a small throughput cost; at batch 5 that trade may be worthwhile if it
converts an OOM into a valid run.
