---
title: "Qwen3 JAX v102 - shard acts s8k bs3 tokamax splash layeragg no scheduler memory pressure"
type: experiment
hypothesis: "Disabling scheduler memory-pressure tracking on the v090 frontier may recover schedule freedom without changing model semantics."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, latency-hiding, memory-pressure, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v102 - shard acts s8k bs3 tokamax splash layeragg no scheduler memory pressure

## Hypothesis under test

**Hypothesis**: The v090 frontier carries
`--xla_tpu_enable_scheduler_memory_pressure_tracking=true` from the inherited
scheduler bundle. v093 and v101 both show that lower-HBM schedules can be
slower, so memory-pressure-aware scheduling may be over-constraining this graph
after scoped VMEM, selective resources, and the layer-scheduler/sparse-core pair
are active.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_tpu_enable_scheduler_memory_pressure_tracking=true` to `false`.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or the profile shows a
useful reduction in all-gather/data-formatting/loop-fusion time without
increasing peak HBM beyond capacity.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v102-nomemtrack`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v102-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-scheduler-memory-pressure`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: v090 scheduler frontier, except
  `--xla_tpu_enable_scheduler_memory_pressure_tracking=false`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v102-nomemtrack` completed cleanly on 2026-06-04 with
valid loss, but did not exceed the v090 frontier band.

- Worker 0: 51,019 tok/s, 6,377 tok/s/chip, 36.6% MFU
- Worker 1: 50,997 tok/s, 6,375 tok/s/chip, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is valid and in-band, but below v090's 51,033 best observed and below the
51,026+ repeated v089/v090 band.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v102-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-scheduler-memory-pressure/2026_06_04_08_08_33`

Profile:

- Total time: 92,773.1 ms; idle 285.9 ms / 0.3%.
- `jit_train_step`: 92,486.6 ms / 99.7%.
- Convolution fusion: 45,719.2 ms / 49.3%.
- Custom-call: 22,055.7 ms / 23.8%.
  - `splash_mha_dkv_no_residuals.1`: 11,641.1 ms / 12.5%.
  - `splash_mha_fwd_residuals.2/3`: 10,414.6 ms / 11.2%.
- Loop fusion: 16,710.9 ms / 18.0%.
- Data formatting: 3,760.8 ms / 4.1%.
- Collective-permute-start: 930.0 ms / 1.0%.
- Peak HBM: 28.32 GiB of 31.25 GiB, with 2.93 GiB free.

The profile is effectively the same as v089/v090; disabling memory-pressure
tracking does not reveal a better schedule.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v102-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-scheduler-memory-pressure/hlo/`

- Object count: 1,839.
- Train-step optimized HLO:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Train-step optimized text size: 1,375,271 bytes, matching v089/v090.
- Pod command confirms
  `--xla_tpu_enable_scheduler_memory_pressure_tracking=false`.

## Verdict

**Refuted / tie.** Disabling scheduler memory-pressure tracking is valid but
does not improve throughput or profile shape. Keep
`--xla_tpu_enable_scheduler_memory_pressure_tracking=true` in the frontier
stack.
