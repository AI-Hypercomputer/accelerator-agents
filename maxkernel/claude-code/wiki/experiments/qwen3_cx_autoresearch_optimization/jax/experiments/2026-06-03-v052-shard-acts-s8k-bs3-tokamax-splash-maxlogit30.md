---
title: "Qwen3 JAX v052 - shard acts s8k bs3 tokamax splash maxlogit30"
type: experiment
hypothesis: "Tokamax Splash max_logit_const may improve the valid v045 attention path without q-seq sharding."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, remat, maxtext-ce, activation-sharding, long-seq, supported, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v052 - shard acts s8k bs3 tokamax splash maxlogit30

## Hypothesis under test

**Hypothesis**: Setting `TOKAMAX_MAX_LOGIT_CONST=30.0` alone may improve the
valid v045 tokamax Splash path without introducing the q-seq NaNs from
v050/v051.

**Mechanism**: Keep v045 fixed (`USE_TOKAMAX_SPLASH=1`, `q_seq_shards=1`,
base2 exp enabled, fused reciprocal enabled, experimental scheduler disabled)
and set only `TOKAMAX_MAX_LOGIT_CONST=30.0`.

**Predicted signal**: Loss stays near the v045/v049 trajectory, ending around
12.05, and throughput exceeds v045's 49,069 tok/s / 35.2% MFU.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v045's 49,069 tok/s / 35.2% MFU.

## Setup

GKE workload `alekseyv-qwen3-v052`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v052-shard-acts-s8k-bs3-tokamax-splash-maxlogit30`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run and new target-sequence frontier:

- Worker 0: 50,376 tok/s total, 6,297 tok/s/chip, 36.1% MFU.
- Worker 1: 50,345 tok/s total, 6,293 tok/s/chip, 36.1% MFU.
- Step time stabilized around 3.90 s after warmup.
- Loss trajectory matched v045/v049, ending at 12.0508.

Compared with v045's 49,069 tok/s / 35.2% MFU, `max_logit_const=30.0`
improves throughput by 1,276-1,307 tok/s, about 2.6-2.7%, and improves MFU by
0.9 points.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v052-shard-acts-s8k-bs3-tokamax-splash-maxlogit30/2026_06_03_04_21_56`
- **MXU utilization**: 55.9%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.4%.

Top device buckets:

- Convolution fusion: 48.1%.
- Custom-call: 23.4%.
- Loop fusion: 18.5%.
- Data formatting: 4.0%.

Compared with v045, custom-call falls from 25.5% to 23.4%. The main win is the
forward residual custom-call path falling from about 13.5% to 11.1%. DKV is
roughly flat/slightly higher at 12.3%.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v052-shard-acts-s8k-bs3-tokamax-splash-maxlogit30/hlo/`
- **Size**: 19.02 MiB
- **Object count**: 26

## Verdict

**Supported.** `TOKAMAX_MAX_LOGIT_CONST=30.0` on the valid v045 tokamax Splash
path is a new target-sequence frontier at 50,376 tok/s / 36.1% MFU.
