---
title: "Qwen3 JAX v056 - shard acts s8k bs3 tokamax splash maxlogit30 rerun3"
type: experiment
hypothesis: "An extra XLA latency-hiding scheduler rerun may improve the v052 frontier schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, remat, maxtext-ce, activation-sharding, long-seq, supported, frontier, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v056 - shard acts s8k bs3 tokamax splash maxlogit30 rerun3

## Hypothesis under test

**Hypothesis**: Increasing `--xla_latency_hiding_scheduler_rerun` from 2 to 3
may improve the v052 frontier schedule.

**Mechanism**: Keep v052 fixed (`TOKAMAX_MAX_LOGIT_CONST=30.0`, target shape,
scan/remat/Splash/MaxText CE/activation sharding) and change only the XLA
latency-hiding scheduler rerun count in `LIBTPU_INIT_ARGS`.

**Predicted signal**: Loss stays near the v052 trajectory, ending around
12.05, and throughput exceeds v052's 50,376 tok/s / 36.1% MFU.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v052's 50,376 tok/s / 36.1% MFU.

## Setup

GKE workload `alekseyv-qwen3-v056`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v056-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-rerun3`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: `--xla_latency_hiding_scheduler_rerun=3` instead of v052's `2`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run and marginal new throughput high-water mark:

- Worker 0: 50,383 tok/s total, 6,298 tok/s/chip, 36.1% MFU.
- Worker 1: 50,356 tok/s total, 6,294 tok/s/chip, 36.1% MFU.
- Step time stabilized around 3.90 s after warmup.
- Loss trajectory stayed normal, ending at 12.0508.

Compared with v052's 50,376 tok/s / 36.1% MFU, rerun3 is only +7 tok/s on
worker 0 and +11 tok/s on worker 1. Treat this as a supported but very small
frontier movement rather than a strong effect.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v056-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-rerun3/2026_06_03_04_48_42`
- **MXU utilization**: 56.0%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 48.1%.
- Custom-call: 23.4%.
- Loop fusion: 18.5%.
- Data formatting: 4.0%.

The profile is effectively unchanged from v052. Custom-call remains split
between about 11.1% forward residuals and 12.3% DKV backward.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v056-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-rerun3/hlo/`
- **Size**: 31.79 MiB
- **Object count**: 1286

The object count is much larger than v052 because the rerun3 probe dumped many
compiler-pass artifacts; the final graph profile is otherwise similar.

## Verdict

**Supported, marginal.** `--xla_latency_hiding_scheduler_rerun=3` edges out
v052 in raw throughput while keeping the same rounded MFU. Follow up with
rerun4 once to determine whether this is a real scheduler trend or noise.
