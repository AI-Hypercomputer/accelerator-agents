---
title: "Qwen3 JAX v103 - shard acts s8k bs3 tokamax splash layeragg no all experimental scheduler"
type: experiment
hypothesis: "Disabling the broad all-experimental scheduler feature bundle while preserving explicit frontier flags may remove hidden scheduler behavior that is not helping the v090 stack."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v103 - shard acts s8k bs3 tokamax splash layeragg no all experimental scheduler

## Hypothesis under test

**Hypothesis**: The v090 frontier carries
`--xla_tpu_enable_all_experimental_scheduler_features=true` in addition to the
explicitly supported flags that were individually adopted. The broad bundle may
enable hidden scheduler behavior that is not necessary after the frontier flags
are specified directly.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_tpu_enable_all_experimental_scheduler_features=true` to `false`. Keep
explicit scoped VMEM, custom-call VMEM adjustment, selective resources,
layer scheduler, sparse-core collective aggregator, memory-pressure tracking,
AG backward pipelining, async collective fusion, and scheduler rerun3 enabled.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or the profile shows a
useful reduction in loop fusion / collective overhead.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v103-noallsched`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v103-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-all-experimental-scheduler`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: v090 scheduler frontier, except
  `--xla_tpu_enable_all_experimental_scheduler_features=false`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v103-noallsched` completed cleanly on 2026-06-04 with
valid loss, but did not exceed the v090 frontier band.

- Worker 0: 51,004 tok/s, 6,375 tok/s/chip, 36.6% MFU
- Worker 1: 50,971 tok/s, 6,371 tok/s/chip, 36.5% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is below v090's 51,033 best observed and below the 51,026+ repeated
v089/v090 band.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v103-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-all-experimental-scheduler/2026_06_04_08_18_27`

Profile:

- Total time: 92,877.8 ms; idle 342.5 ms / 0.4%.
- `jit_train_step`: 92,534.7 ms / 99.6%.
- Convolution fusion: 45,721.9 ms / 49.2%.
- Custom-call: 22,055.9 ms / 23.7%.
  - `splash_mha_dkv_no_residuals.1`: 11,641.1 ms / 12.5%.
  - `splash_mha_fwd_residuals.2/3`: 10,414.8 ms / 11.2%.
- Loop fusion: 16,712.9 ms / 18.0%.
- Data formatting: 3,768.9 ms / 4.1%.
- Collective-permute-start: 931.0 ms / 1.0%.
- Peak HBM: 28.32 GiB of 31.25 GiB, with 2.93 GiB free.

The profile matches v102 and is effectively in-band with v089/v090, but without
a throughput gain.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v103-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-all-experimental-scheduler/hlo/`

- Object count: 1,839.
- Train-step optimized HLO:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Train-step optimized text size: 1,375,271 bytes, matching v089/v090/v102.
- Pod command confirms
  `--xla_tpu_enable_all_experimental_scheduler_features=false`.

## Verdict

**Refuted / tie.** Disabling the broad all-experimental scheduler feature
bundle while keeping the explicit adopted frontier flags is valid, but it does
not improve throughput or profile shape. Keep
`--xla_tpu_enable_all_experimental_scheduler_features=true` in the operational
frontier unless a later simplification pass removes it for cleanliness.
