---
title: "Qwen3 JAX v058 - shard acts s8k bs3 tokamax splash maxlogit30 cost estimates"
type: experiment
hypothesis: "Explicit Tokamax Splash cost estimates may improve scheduling of the remaining attention custom calls."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, cost-model, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v058 - shard acts s8k bs3 tokamax splash maxlogit30 cost estimates

## Hypothesis under test

**Hypothesis**: Supplying explicit Tokamax/Pallas forward and backward FLOP
cost estimates may improve XLA scheduling of the remaining custom-call bucket.

**Mechanism**: Keep the v056 frontier fixed (`TOKAMAX_MAX_LOGIT_CONST=30.0`,
target shape, scan/remat/Splash/MaxText CE/activation sharding, scheduler
rerun3) and set only `TOKAMAX_COST_FWD=3.3e12 TOKAMAX_COST_BWD=5.2e12`. These
are approximate per-call costs inferred from the v057 XProf custom-call FLOP
totals.

**Predicted signal**: Loss stays near the v052-v057 trajectory, ending around
12.05, and throughput exceeds v056's 50,383 tok/s / 36.1% MFU or reduces the
23.4% custom-call bucket without a wall-clock regression.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU
without a meaningful custom-call reduction.

## Setup

GKE workload `alekseyv-qwen3-v058`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v058-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-cost-est`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_COST_FWD=3300000000000 TOKAMAX_COST_BWD=5200000000000`
- **XLA flag**: keep v056's `--xla_latency_hiding_scheduler_rerun=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but still below v056:

- Worker 0: 50,357 tok/s total, 6,295 tok/s/chip, 36.1% MFU.
- Worker 1: 50,379 tok/s total, 6,297 tok/s/chip, 36.1% MFU.
- Step time stabilized around 3.90 s after warmup.
- Loss trajectory stayed normal, ending at 12.0508.

Compared with v056's 50,383 tok/s / 36.1% MFU, explicit cost estimates do not
improve wall-clock throughput. Worker1 is close, but worker0 and the profile
do not support a new frontier.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v058-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-cost-est/2026_06_03_20_39_11`
- **MXU utilization**: 56.0%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 48.1%.
- Custom-call: 23.4%.
- Loop fusion: 18.6%.
- Data formatting: 4.0%.

Custom-call time is unchanged: DKV backward remains about 12.3%, and forward
residuals remain about 11.1%. The cost hints affect XProf FLOP accounting for
custom calls, but not observed device time.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v058-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-cost-est/hlo/`
- **Size**: 19.03 MiB
- **Object count**: 32

## Verdict

**Refuted.** Explicit Tokamax/Pallas cost hints with the measured per-call
FLOP scale do not reduce the attention custom-call bucket or beat the v056
throughput frontier.
