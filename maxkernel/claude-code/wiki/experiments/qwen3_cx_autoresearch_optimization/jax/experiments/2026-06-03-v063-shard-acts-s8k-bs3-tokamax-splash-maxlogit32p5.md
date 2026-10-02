---
title: "Qwen3 JAX v063 - shard acts s8k bs3 tokamax splash maxlogit32p5"
type: experiment
hypothesis: "A max-logit bound between 30 and 35 may edge out the v056 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, max-logit, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v063 - shard acts s8k bs3 tokamax splash maxlogit32p5

## Hypothesis under test

**Hypothesis**: `TOKAMAX_MAX_LOGIT_CONST=32.5` may sit closer to the narrow
max-logit optimum than 35.0 while preserving the v056 gain over v045.

**Mechanism**: Keep the v056 frontier fixed (target shape, scan/remat/Splash,
MaxText CE, activation sharding, scheduler rerun3) and change only the
Tokamax Splash max-logit bound from 30.0 to 32.5.

**Predicted signal**: Loss stays near the v052-v062 trajectory, ending around
12.05, and throughput exceeds v056's 50,383 tok/s / 36.1% MFU.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU.

## Setup

GKE workload `alekseyv-qwen3-v063`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v063-shard-acts-s8k-bs3-tokamax-splash-maxlogit32p5`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=32.5`
- **XLA flag**: keep v056's `--xla_latency_hiding_scheduler_rerun=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but it does not exceed the v056 frontier:

- Worker 0: 50,383 tok/s total, 6,298 tok/s/chip, 36.1% MFU.
- Worker 1: 50,361 tok/s total, 6,295 tok/s/chip, 36.1% MFU.
- Step time stabilized around 3.90 s after warmup.
- Loss trajectory stayed normal, ending at 12.0507.

Compared with v056's 50,383 tok/s / 36.1% MFU, `max_logit_const=32.5`
matches the rounded/raw displayed worker-0 throughput but does not improve it;
worker 1 remains below the frontier.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v063-shard-acts-s8k-bs3-tokamax-splash-maxlogit32p5/2026_06_03_21_12_06`
- **MXU utilization**: 56.0%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 48.2%.
- Custom-call: 23.4%.
- Loop fusion: 18.5%.
- Data formatting: 4.0%.

The profile is unchanged from the v056/v060 shape. Custom-call remains split
between about 11.1% forward residuals and 12.3% DKV backward.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v063-shard-acts-s8k-bs3-tokamax-splash-maxlogit32p5/hlo/`
- **Size**: 18.71 MiB
- **Object count**: 28

## Verdict

**Refuted, tie.** `TOKAMAX_MAX_LOGIT_CONST=32.5` is valid and ties the v056
worker-0 display, but it does not create a new throughput frontier or move the
profile. Treat `TOKAMAX_MAX_LOGIT_CONST=30.0` plus scheduler rerun3 as the
formal frontier and pivot away from narrow max-logit sweeps.
