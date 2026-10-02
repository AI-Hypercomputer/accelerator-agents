---
title: "Qwen3 JAX v114 - shard acts s8k bs3 tokamax splash layeragg no acf reducescatter"
type: experiment
hypothesis: "Async collective fusion's reduce-scatter sub-pass may be neutral or harmful under the v090 layeragg scheduler while base async collective fusion and all-gather fusion remain useful."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, async-collective-fusion, reduce-scatter, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v114 - shard acts s8k bs3 tokamax splash layeragg no acf reducescatter

## Hypothesis under test

**Hypothesis**: v113 showed the all-gather async collective fusion subtype is
required. The reduce-scatter subtype is still unisolated. Under the
layer-scheduler/sparse-core-aggregator pair, keeping base async collective
fusion and all-gather fusion enabled while disabling reduce-scatter fusion may
remove a scheduling constraint without reintroducing the all-gather bucket.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true` to `false`.
Leave base async collective fusion, all-gather fusion, and multiple-steps
fusion enabled.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or XProf shows lower
train-step wall time without reintroducing all-gather or reduce-scatter cost.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v114-noacfrs`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v114-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-reducescatter`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v090 scheduler frontier, except
  `--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v114-noacfrs` completed cleanly on
`<your-cluster>`.

- Worker 0: 51,008 tok/s average, 36.6% MFU.
- Worker 1: 50,982 tok/s average, 36.6% MFU.
- Final loss: 12.0508 on both workers.
- XProf run:
  `2026-06-04-qwen3-jax-v114-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-reducescatter/2026_06_04_10_24_55`.

This is valid but below both the v090 best-observed 51,033 tok/s and the
51,026+ repeated v089/v090 frontier band.

## Profile

XProf shows this is effectively the normal layeragg frontier profile rather
than a beneficial schedule change.

- Total profiled time: 92,807.7 ms.
- MXU utilization: 56.8%.
- Peak HBM: 28.32 GiB.
- Custom-call: 22,055.9 ms / 23.8%.
- DKV custom-call: 11,641.1 ms / 12.5%.
- Forward residual custom-calls: 5,213.0 ms / 5.6% and 5,201.9 ms / 5.6%.
- Convolution fusion: 45,717.9 ms / 49.3%.
- Loop fusion: 16,723.6 ms / 18.0%.
- Data formatting: 3,773.3 ms / 4.1%.
- Collective-permute-start: 922.3 ms / 1.0%.

Unlike v113, this does not expose an all-gather bucket. The result is simply
below the repeated frontier band with no profile win.

## HLO Dump

Train-step HLO artifacts were written under the run's `hlo/` prefix.

- Optimized train-step text:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 1,375,271 bytes.
- HLO object count under the run prefix: 2,210.
- Content hash: `796167499c5335993c3c6cc63486cb24875556369b39d8bdc1e70d30ebeb969c`.
- XProf train-step program hash: `13194376450853388418`.

The train-step program hash matches the normal v090/v110 frontier family.

## Verdict

Refuted/tie. Disabling the reduce-scatter async collective fusion subtype is
valid but does not improve the v090 layeragg frontier. Keep
`--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true`.
