---
title: "Qwen3 JAX v110 - shard acts s8k bs3 tokamax splash layeragg rerun4"
type: experiment
hypothesis: "Increasing latency-hiding scheduler reruns on the layeragg frontier may find a better valid overlap schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, latency-hiding, scheduler-rerun, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v110 - shard acts s8k bs3 tokamax splash layeragg rerun4

## Hypothesis under test

**Hypothesis**: v057 showed `--xla_latency_hiding_scheduler_rerun=4` does not
improve the older v056 max-logit frontier. The current v090 frontier adds
selective resources, scoped VMEM adjustment, latency-hiding layer scheduler, and
sparse-core collective aggregation. Rerun4 may interact with this fuller
scheduler stack and find a better overlap schedule than rerun3.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_latency_hiding_scheduler_rerun=3` to `4`.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or the profile shows a
useful reduction in custom-call, DKV, loop-fusion, data-formatting, or idle time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v110-rerun4`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v110-shard-acts-s8k-bs3-tokamax-splash-layeragg-rerun4`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: v090 scheduler frontier, except
  `--xla_latency_hiding_scheduler_rerun=4`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v110-rerun4` completed cleanly on 2026-06-04 with
normal loss, but did not improve the v089/v090 frontier.

- **Worker 0 pod**: `alekseyv-qwen3-v110-rerun4-slice-job-0-0-6742z`
- **Worker 0 throughput**: 51,009 tok/s, 6,376 tok/s/chip
- **Worker 0 MFU**: 36.6%
- **Worker 1 pod**: `alekseyv-qwen3-v110-rerun4-slice-job-0-1-t9mqc`
- **Worker 1 throughput**: 50,992 tok/s, 6,374 tok/s/chip
- **Worker 1 MFU**: 36.6%
- **Final loss**: 12.0508
- **Exit status**: 0 on both workers

This is valid, but below v090's 51,033 tok/s best-observed worker and below the
51,026+ tok/s repeated v089/v090 band.

## Profile

- **XProf run**: `2026-06-04-qwen3-jax-v110-shard-acts-s8k-bs3-tokamax-splash-layeragg-rerun4/2026_06_04_09_35_42`
- **Total time**: 92,743.683 ms
- **Idle**: 227.307 ms / 0.2%
- **Compute**: 92,516.376 ms
- **Top program**: `jit_train_step(13194376450853388418)`, 92,515.694 ms / 99.8%
- **Convolution fusion**: 45,719.556 ms / 49.3%
- **Custom-call**: 22,056.663 ms / 23.8%
- **DKV backward bucket**: 11,641.093 ms / 12.6%
- **Forward residual custom-calls**: 5,213.719 ms + 5,201.844 ms / 11.2%
- **Loop fusion**: 16,721.228 ms / 18.0%
- **Data formatting**: 3,774.001 ms / 4.1%
- **Peak HBM**: 28.32 GiB, with 2.9278 GiB free
- **Step average**: 1657.7 ms
- **MXU utilization**: 56.9%

The profile is the normal valid-frontier shape: custom-call remains about 23.8%,
DKV about 12.6%, loop fusion about 18%, and data formatting about 4.1%.

## HLO Dump

The run wrote the normal train-step HLO:

- **Optimized train-step HLO**: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **Train-step HLO size**: 1,375,271 bytes
- **HLO object count**: 1,619
- **Program hash**: `13194376450853388418`

This matches the v089/v090/v102-v109 compiled program family.

## Verdict

Refuted/tie. Increasing `--xla_latency_hiding_scheduler_rerun` from 3 to 4 is
valid on the layeragg frontier, but it leaves the HLO/profile shape unchanged
and reaches only 51,009 tok/s best worker. Keep
`--xla_latency_hiding_scheduler_rerun=3`.
