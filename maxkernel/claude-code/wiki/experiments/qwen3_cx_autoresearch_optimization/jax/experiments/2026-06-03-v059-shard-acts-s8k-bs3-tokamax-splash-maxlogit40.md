---
title: "Qwen3 JAX v059 - shard acts s8k bs3 tokamax splash maxlogit40"
type: experiment
hypothesis: "A looser Tokamax Splash max-logit bound may improve the v056 frontier schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, max-logit, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v059 - shard acts s8k bs3 tokamax splash maxlogit40

## Hypothesis under test

**Hypothesis**: Increasing `TOKAMAX_MAX_LOGIT_CONST` from 30.0 to 40.0 may
preserve the v052/v056 max-logit scheduling win while slightly improving the
remaining attention custom-call balance.

**Mechanism**: Keep the v056 frontier fixed (target shape, scan/remat/Splash,
MaxText CE, activation sharding, scheduler rerun3) and change only the
Tokamax Splash max-logit bound.

**Predicted signal**: Loss stays near the v052-v058 trajectory, ending around
12.05, and throughput exceeds v056's 50,383 tok/s / 36.1% MFU or reduces the
23.4% custom-call bucket without a wall-clock regression.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU
without a meaningful custom-call reduction.

## Setup

GKE workload `alekseyv-qwen3-v059`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v059-shard-acts-s8k-bs3-tokamax-splash-maxlogit40`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=40.0`
- **XLA flag**: keep v056's `--xla_latency_hiding_scheduler_rerun=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but below v056:

- Worker 0: 50,350 tok/s total, 6,294 tok/s/chip, 36.1% MFU.
- Worker 1: 50,374 tok/s total, 6,297 tok/s/chip, 36.1% MFU.
- Step time stabilized around 3.90 s after warmup.
- Loss trajectory stayed normal, ending at 12.0506.

Compared with v056's 50,383 tok/s / 36.1% MFU, `max_logit_const=40.0` is a
small but consistent regression.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v059-shard-acts-s8k-bs3-tokamax-splash-maxlogit40/2026_06_03_20_49_31`
- **MXU utilization**: 55.9%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 48.1%.
- Custom-call: 23.4%.
- Loop fusion: 18.6%.
- Data formatting: 4.0%.

The custom-call split is unchanged from v056/v058: DKV backward remains about
12.3%, and forward residuals remain about 11.1%.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v059-shard-acts-s8k-bs3-tokamax-splash-maxlogit40/hlo/`
- **Size**: 19.02 MiB
- **Object count**: 32

## Verdict

**Refuted.** `TOKAMAX_MAX_LOGIT_CONST=40.0` is valid but does not improve over
the v056 frontier. If continuing the max-logit axis, probe closer to 30.0;
otherwise pivot to a structural attention-kernel change.
