---
title: "Qwen3 JAX v106 - shard acts s8k bs3 tokamax splash layeragg shmem90"
type: experiment
hypothesis: "Reducing the TPU scheduler shared-memory limit on the v090 frontier may avoid an over-aggressive schedule without changing model semantics."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, shared-memory, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v106 - shard acts s8k bs3 tokamax splash layeragg shmem90

## Hypothesis under test

**Hypothesis**: The v090 frontier carries
`--xla_tpu_scheduler_percent_shared_memory_limit=100`, allowing the scheduler to
use the full shared-memory budget. Since v093 showed lower-HBM schedules can be
slower but v102-v105 show several host/scheduler toggles are in-band, a modest
shared-memory cap may produce a different schedule that reduces loop fusion or
copy pressure without exceeding HBM.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_tpu_scheduler_percent_shared_memory_limit=100` to `90`.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or the profile shows a
useful reduction in loop-fusion/data-formatting/custom-call time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v106-shmem90`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v106-shard-acts-s8k-bs3-tokamax-splash-layeragg-shmem90`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: v090 scheduler frontier, except
  `--xla_tpu_scheduler_percent_shared_memory_limit=90`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v106-shmem90` completed cleanly on 2026-06-04 with
valid loss, but did not exceed the v090 frontier band.

- Worker 0: 51,000 tok/s, 6,375 tok/s/chip, 36.6% MFU
- Worker 1: 50,970 tok/s, 6,371 tok/s/chip, 36.5% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is below v090's 51,033 best observed and below the 51,026+ repeated
v089/v090 band.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v106-shard-acts-s8k-bs3-tokamax-splash-layeragg-shmem90/2026_06_04_08_50_51`

Profile:

- Total time: 92,816.9 ms; idle 293.3 ms / 0.3%.
- `jit_train_step`: 92,523.0 ms / 99.7%.
- Convolution fusion: 45,707.2 ms / 49.2%.
- Custom-call: 22,077.5 ms / 23.8%.
  - `splash_mha_dkv_no_residuals.1`: 11,658.1 ms / 12.6%.
  - `splash_mha_fwd_residuals.2/3`: 10,419.3 ms / 11.2%.
- Loop fusion: 16,778.7 ms / 18.1%.
- Data formatting: 3,776.1 ms / 4.1%.
- Peak HBM: 28.15 GiB of 31.25 GiB, with 3.09 GiB free.

The shared-memory cap produced a real lower-HBM schedule, but the wall time
regressed and custom-call / loop-fusion time did not improve.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v106-shard-acts-s8k-bs3-tokamax-splash-layeragg-shmem90/hlo/`

- Train-step optimized HLO:
  `module_0264.jit_train_step.cl_854318611.after_optimizations.txt` and
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Train-step optimized text size: 1,375,670 bytes, larger than the
  1,375,271-byte v089/v090/v102-v105 train-step HLO.
- XProf train-step program hash changed to `798154806117746318`.
- Pod command confirms `--xla_tpu_scheduler_percent_shared_memory_limit=90`.

## Verdict

**Refuted / tie.** Reducing the scheduler shared-memory limit from 100 to 90 is
valid and lowers peak HBM by about 0.17 GiB, but it does not improve throughput
or the profile bottlenecks. Keep
`--xla_tpu_scheduler_percent_shared_memory_limit=100` in the operational
frontier for throughput.
