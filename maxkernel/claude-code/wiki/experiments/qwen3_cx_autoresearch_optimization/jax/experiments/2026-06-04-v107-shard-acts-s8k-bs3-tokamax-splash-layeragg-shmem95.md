---
title: "Qwen3 JAX v107 - shard acts s8k bs3 tokamax splash layeragg shmem95"
type: experiment
hypothesis: "A mild TPU scheduler shared-memory cap may preserve v090 throughput while avoiding the slower v106 low-HBM schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, shared-memory, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v107 - shard acts s8k bs3 tokamax splash layeragg shmem95

## Hypothesis under test

**Hypothesis**: v106 showed that
`--xla_tpu_scheduler_percent_shared_memory_limit=90` creates a real lower-HBM
schedule, but it slows throughput. A milder cap at 95 may avoid the slower
schedule corner while still giving the scheduler enough pressure to improve
copy/loop-fusion placement.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_tpu_scheduler_percent_shared_memory_limit=100` to `95`.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or the profile shows a
lower-HBM schedule without the v106 loop-fusion/custom-call regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v107-shmem95`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v107-shard-acts-s8k-bs3-tokamax-splash-layeragg-shmem95`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: v090 scheduler frontier, except
  `--xla_tpu_scheduler_percent_shared_memory_limit=95`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v107-shmem95` completed cleanly on 2026-06-04 with
valid loss, but did not exceed the v090 frontier band.

- Worker 0: 51,012 tok/s, 6,376 tok/s/chip, 36.6% MFU
- Worker 1: 50,988 tok/s, 6,374 tok/s/chip, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is below v090's 51,033 best observed and below the 51,026+ repeated
v089/v090 band.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v107-shard-acts-s8k-bs3-tokamax-splash-layeragg-shmem95/2026_06_04_09_00_58`

Profile:

- Total time: 92,763.4 ms; idle 270.6 ms / 0.3%.
- `jit_train_step`: 92,492.1 ms / 99.7%.
- Convolution fusion: 45,709.2 ms / 49.3%.
- Custom-call: 22,056.0 ms / 23.8%.
  - `splash_mha_dkv_no_residuals.1`: 11,641.1 ms / 12.5%.
  - `splash_mha_fwd_residuals.2/3`: 10,414.9 ms / 11.2%.
- Loop fusion: 16,714.6 ms / 18.0%.
- Data formatting: 3,771.4 ms / 4.1%.
- Peak HBM: 28.32 GiB of 31.25 GiB, with 2.93 GiB free.

Unlike v106's limit-90 run, the limit-95 run returned to the normal frontier
program/profile shape and did not retain the lower-HBM schedule.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v107-shard-acts-s8k-bs3-tokamax-splash-layeragg-shmem95/hlo/`

- Object count: 1,619.
- Train-step optimized HLO:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Train-step optimized text size: 1,375,271 bytes, matching v089/v090/v102-v105.
- XProf train-step program hash returned to `13194376450853388418`.
- Pod command confirms `--xla_tpu_scheduler_percent_shared_memory_limit=95`.

## Verdict

**Refuted / tie.** Reducing the scheduler shared-memory limit from 100 to 95 is
valid, but it does not improve throughput or create the lower-HBM schedule seen
at 90. Keep `--xla_tpu_scheduler_percent_shared_memory_limit=100` in the
operational frontier.
