---
title: "Qwen3 JAX v120 - shard acts s8k bs5 tokamax splash layeragg collective matmul none shmem90"
type: experiment
hypothesis: "Lowering scheduler shared-memory limit can convert the v119 batch-5 near-fit OOM into a valid run."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, shared-memory, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v120 - shard acts s8k bs5 tokamax splash layeragg collective matmul none shmem90

## Hypothesis under test

**Hypothesis**: v119 missed HBM capacity by only 46.98 MiB, so a lower-HBM
scheduler variant can make batch 5/global 40 fit. Earlier bs3 testing showed
`--xla_tpu_scheduler_percent_shared_memory_limit=90` can select a real
lower-HBM schedule, with a small throughput cost.

**Mechanism**: Keep v119 fixed and change only:

```text
--xla_tpu_scheduler_percent_shared_memory_limit=100 -> 90
```

**Predicted signal**: Compile succeeds and the run reaches finite-loss steps. If
valid, total throughput should be compared against v118's 56,445 tok/s / 40.5%
MFU frontier; even a slight per-step slowdown could still win at global batch
40.

**Falsification criterion**: OOM/compile failure, invalid loss, or clean
completion that does not beat v118 throughput.

## Setup

GKE workload `alekseyv-qwen3-v120-bs5-shmem90`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v120-shard-acts-s8k-bs5-tokamax-splash-layeragg-collective-matmul-none-shmem90`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v119 stack, except scheduler shared-memory limit is 90.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=5 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 40
- **Profile window**: steps 12-14

## Results

Failed during train-step compile before any measured step (`EXIT_CODE=1` on both
workers):

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 32.37G of 31.25G hbm.
Exceeded hbm capacity by 1.12G.
```

This is worse than v119's 46.98 MiB near-fit OOM, so the shared-memory cap
selected a lower-throughput/larger-HBM schedule rather than relieving pressure.

## Profile

No runtime profile was captured because the workload failed at compile time.

## HLO Dump

Train-step OOM artifacts were emitted under:

```text
module_0267.jit_train_step.cl_854318611
```

- Optimized train-step text size: 726,125 bytes.
- HLO content hash: `1f3ef62bd5511ce5ac592a076ee65e5755d8f7fa5df3d17a5f2027d79b6a7644`.
- HLO prefix object count: 1,288.
- Flagfile confirms both collective-matmul modes are `NONE`, scoped VMEM is
  81920 KiB, and scheduler shared-memory limit is 90.

## Verdict

**Refuted / invalid.** Lowering scheduler shared-memory limit to 90 does not
make batch 5 fit; it worsens the OOM margin from 46.98 MiB to 1.12 GiB and
regenerates a much larger HLO artifact set. Do not carry the shared-memory cap
for this batch-scaling path.

Next fit attempt should restore shared-memory limit 100 and lower scoped VMEM
instead, because v119 missed by only 46.98 MiB and the v068-v071 VMEM sweep
showed scoped VMEM is a direct schedule-selection surface.
