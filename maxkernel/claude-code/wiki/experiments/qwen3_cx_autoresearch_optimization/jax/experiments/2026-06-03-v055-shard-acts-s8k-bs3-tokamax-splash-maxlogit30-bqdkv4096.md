---
title: "Qwen3 JAX v055 - shard acts s8k bs3 tokamax splash maxlogit30 bqdkv4096"
type: experiment
hypothesis: "A larger DKV query block may reduce backward Splash time on the v052 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v055 - shard acts s8k bs3 tokamax splash maxlogit30 bqdkv4096

## Hypothesis under test

**Hypothesis**: Increasing `SPLASH_BQ_DKV` from 2048 to 4096 on the v052
frontier may reduce DKV overhead without the wall-clock loss seen from
`TOKAMAX_DQ_REDUCTION_STEPS=3`.

**Mechanism**: Keep v052 fixed and set only `SPLASH_BQ_DKV=4096`. Forward
block sizes and DKV KV block sizes remain at the v052 defaults.

**Predicted signal**: Loss stays near the v052 trajectory, ending around
12.05, and throughput exceeds v052's 50,376 tok/s / 36.1% MFU.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v052's 50,376 tok/s / 36.1% MFU.

## Setup

GKE workload `alekseyv-qwen3-v055`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v055-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-bqdkv4096`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0 SPLASH_BQ_DKV=4096`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but a large regression:

- Worker 0: 41,599 tok/s total, 5,200 tok/s/chip, 29.8% MFU.
- Worker 1: 41,581 tok/s total, 5,198 tok/s/chip, 29.8% MFU.
- Step time stabilized around 4.73 s after warmup.
- Loss trajectory stayed normal, ending at 12.0508.

Compared with v052's 50,376 tok/s / 36.1% MFU, `SPLASH_BQ_DKV=4096` costs
about 17%.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v055-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-bqdkv4096/2026_06_03_04_39_08`
- **MXU utilization**: 48.0%.
- **Peak HBM**: 28.33 GiB, 2.912 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 39.7%.
- Custom-call: 36.8%.
- Loop fusion: 15.3%.
- Data formatting: 3.3%.

The regression is entirely in backward Splash: DKV custom-call rises to 27.6%
versus v052's 12.3%. Forward residuals remain around 9.2%.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v055-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-bqdkv4096/hlo/`
- **Size**: 19.02 MiB
- **Object count**: 26

## Verdict

**Refuted.** Larger DKV query blocks are the wrong direction at this shape;
`SPLASH_BQ_DKV=4096` makes the DKV kernel dominate runtime.
