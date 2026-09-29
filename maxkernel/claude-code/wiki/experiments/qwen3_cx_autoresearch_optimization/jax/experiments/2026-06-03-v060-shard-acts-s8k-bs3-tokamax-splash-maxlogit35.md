---
title: "Qwen3 JAX v060 - shard acts s8k bs3 tokamax splash maxlogit35"
type: experiment
hypothesis: "A max-logit bound between 30 and 40 may recover the v056 frontier or expose a narrow optimum."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, max-logit, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v060 - shard acts s8k bs3 tokamax splash maxlogit35

## Hypothesis under test

**Hypothesis**: `TOKAMAX_MAX_LOGIT_CONST=35.0` may be closer to the local
max-logit optimum than the refuted 40.0 while preserving the v052/v056 gain.

**Mechanism**: Keep the v056 frontier fixed (target shape, scan/remat/Splash,
MaxText CE, activation sharding, scheduler rerun3) and change only the
Tokamax Splash max-logit bound from 30.0 to 35.0.

**Predicted signal**: Loss stays near the v052-v059 trajectory, ending around
12.05, and throughput exceeds v056's 50,383 tok/s / 36.1% MFU or reduces the
23.4% custom-call bucket without a wall-clock regression.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU
without a meaningful custom-call reduction.

## Setup

GKE workload `alekseyv-qwen3-v060`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v060-shard-acts-s8k-bs3-tokamax-splash-maxlogit35`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=35.0`
- **XLA flag**: keep v056's `--xla_latency_hiding_scheduler_rerun=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but still just below v056:

- Worker 0: 50,381 tok/s total, 6,298 tok/s/chip, 36.1% MFU.
- Worker 1: 50,358 tok/s total, 6,295 tok/s/chip, 36.1% MFU.
- Step time stabilized around 3.90 s after warmup.
- Loss trajectory stayed normal, ending at 12.0510.

Compared with v056's 50,383 tok/s / 36.1% MFU, `max_logit_const=35.0` is an
effective tie but not a new raw-throughput frontier.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v060-shard-acts-s8k-bs3-tokamax-splash-maxlogit35/2026_06_03_20_55_19`
- **MXU utilization**: 56.0%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 48.2%.
- Custom-call: 23.4%.
- Loop fusion: 18.5%.
- Data formatting: 4.0%.

The custom-call split is unchanged: DKV backward remains about 12.3%, and
forward residuals remain about 11.1%.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v060-shard-acts-s8k-bs3-tokamax-splash-maxlogit35/hlo/`
- **Size**: 19.02 MiB
- **Object count**: 32

## Verdict

**Refuted, marginal.** `TOKAMAX_MAX_LOGIT_CONST=35.0` is valid and essentially
ties the frontier, but it does not beat v056 or reduce the custom-call bucket.
Treat 30.0 as the current best max-logit bound.
