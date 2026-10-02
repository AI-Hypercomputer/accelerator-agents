---
title: "Qwen3 JAX v111 - shard acts s8k bs3 tokamax splash layeragg vmem86016"
type: experiment
hypothesis: "A narrower upper-side scoped-VMEM point may interact with the layeragg scheduler stack and improve the valid frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, vmem, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v111 - shard acts s8k bs3 tokamax splash layeragg vmem86016

## Hypothesis under test

**Hypothesis**: v068 established `--xla_tpu_scoped_vmem_limit_kib=81920` as the
best pre-layeragg VMEM point, while v070 showed 90112 was too high before the
layer scheduler and sparse-core collective aggregator were enabled. A narrower
upper-side point, 86016 KiB, may give the current layeragg scheduler stack more
freedom without falling into v070's slower schedule.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_tpu_scoped_vmem_limit_kib=81920` to `86016`.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or the profile shows a
useful drop in loop fusion, custom-call, data formatting, or idle time without
raising HBM pressure.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v111-vmem86016`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v111-shard-acts-s8k-bs3-tokamax-splash-layeragg-vmem86016`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: v090 scheduler frontier, except
  `--xla_tpu_scoped_vmem_limit_kib=86016`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v111-vmem86016` completed cleanly on 2026-06-04
with normal loss, but regressed well below the v089/v090 frontier.

- **Worker 0 pod**: `alekseyv-qwen3-v111-vmem86016-slice-job-0-0-8szgr`
- **Worker 0 throughput**: 50,003 tok/s, 6,250 tok/s/chip
- **Worker 0 MFU**: 35.9%
- **Worker 1 pod**: `alekseyv-qwen3-v111-vmem86016-slice-job-0-1-6frqx`
- **Worker 1 throughput**: 49,982 tok/s, 6,248 tok/s/chip
- **Worker 1 MFU**: 35.8%
- **Final loss**: 12.0511
- **Exit status**: 0 on both workers

The run is valid, but it is roughly 1,030 tok/s below v090's 51,033 tok/s
best-observed worker and clearly below the repeated v089/v090 band.

## Profile

- **XProf run**: `2026-06-04-qwen3-jax-v111-shard-acts-s8k-bs3-tokamax-splash-layeragg-vmem86016/2026_06_04_09_48_25`
- **Total time**: 94,659.446 ms
- **Idle**: 260.795 ms / 0.3%
- **Compute**: 94,398.651 ms
- **Top program**: `jit_train_step(3111414773617791261)`, 94,397.967 ms / 99.7%
- **Convolution fusion**: 45,829.130 ms / 48.4%
- **Custom-call**: 22,006.291 ms / 23.2%
- **DKV backward bucket**: 11,597.093 ms / 12.3%
- **Forward residual custom-calls**: 10,409.190 ms / 11.0%
- **Loop fusion**: 17,924.142 ms / 18.9%
- **Data formatting**: 3,829.042 ms / 4.0%
- **Peak HBM**: 28.32 GiB, with 2.9235 GiB free
- **Step average**: 1692.3 ms
- **MXU utilization**: 55.7%

The profile shows the regression as a scheduler-level wall-time loss rather
than an invalid kernel: loop fusion grows from about 18.0% to 18.9%, total
profile time rises by about 1.9 s versus v110/v090-family runs, and MXU falls
from 56.8-56.9% to 55.7%.

## HLO Dump

The run selected a different train-step program:

- **Optimized train-step HLO**: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **Train-step HLO size**: 1,376,311 bytes
- **HLO object count**: 1,615
- **Program hash**: `3111414773617791261`

This is a real VMEM-driven schedule change, but the new schedule is slower.

## Verdict

Refuted. Raising scoped VMEM from 81920 KiB to 86016 KiB on the layeragg frontier
is valid and changes the compiled program, but it regresses throughput to about
50.0k tok/s and lowers MXU. Keep `--xla_tpu_scoped_vmem_limit_kib=81920`; the
upper VMEM axis remains exhausted for throughput.
