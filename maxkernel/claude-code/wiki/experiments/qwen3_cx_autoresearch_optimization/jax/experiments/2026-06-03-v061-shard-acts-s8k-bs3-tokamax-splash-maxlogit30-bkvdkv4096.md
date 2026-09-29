---
title: "Qwen3 JAX v061 - shard acts s8k bs3 tokamax splash maxlogit30 bkvdkv4096"
type: experiment
hypothesis: "A larger DKV KV block may reduce backward Splash overhead without the BQ_DKV=4096 regression."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, dkv, block-size, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v061 - shard acts s8k bs3 tokamax splash maxlogit30 bkvdkv4096

## Hypothesis under test

**Hypothesis**: Increasing only the DKV KV block from 2048 to 4096 may reduce
the remaining DKV backward custom-call overhead without the hard regression
seen when v055 increased the DKV query block.

**Mechanism**: Keep the v056 frontier fixed (`TOKAMAX_MAX_LOGIT_CONST=30.0`,
target shape, scan/remat/Splash/MaxText CE/activation sharding, scheduler
rerun3) and set only `SPLASH_BKV_DKV=4096 SPLASH_BKV_DKV_COMPUTE=4096`.
Leave `SPLASH_BQ_DKV=2048`.

**Predicted signal**: Loss stays near the v052-v060 trajectory, ending around
12.05, and throughput exceeds v056's 50,383 tok/s / 36.1% MFU or reduces the
12.3% DKV bucket without a wall-clock regression.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU
without a meaningful DKV reduction.

## Setup

GKE workload `alekseyv-qwen3-v061`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v061-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-bkvdkv4096`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0 SPLASH_BKV_DKV=4096 SPLASH_BKV_DKV_COMPUTE=4096`
- **XLA flag**: keep v056's `--xla_latency_hiding_scheduler_rerun=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but a large regression:

- Worker 0: 41,572 tok/s total, 5,196 tok/s/chip, 29.8% MFU.
- Worker 1: 41,587 tok/s total, 5,198 tok/s/chip, 29.8% MFU.
- Step time stabilized around 4.73 s after warmup.
- Loss trajectory stayed normal, ending at 12.0511.

Compared with v056's 50,383 tok/s / 36.1% MFU, increasing the DKV KV block to
4096 costs about 17.5%.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v061-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-bkvdkv4096/2026_06_03_21_01_25`
- **MXU utilization**: 48.0%.
- **Peak HBM**: 28.33 GiB, 2.912 GiB free.
- **Device idle**: 0.2%.

Top device buckets:

- Convolution fusion: 39.7%.
- Custom-call: 37.1%.
- Loop fusion: 15.3%.
- Data formatting: 3.3%.

The regression is entirely in backward Splash: DKV rises from v056's 12.3% to
27.9%. Forward residuals fall in share to about 9.2% only because DKV becomes
dominant.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v061-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-bkvdkv4096/hlo/`
- **Size**: 19.02 MiB
- **Object count**: 32

## Verdict

**Refuted.** Larger DKV KV blocks are the wrong direction at this shape, just
like v055's larger DKV query block. Keep the DKV block sizes at the defaults.
