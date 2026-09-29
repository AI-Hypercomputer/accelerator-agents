---
title: "Qwen3 JAX v053 - shard acts s8k bs3 tokamax splash maxlogit20"
type: experiment
hypothesis: "A tighter tokamax Splash max_logit_const may improve the v052 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v053 - shard acts s8k bs3 tokamax splash maxlogit20

## Hypothesis under test

**Hypothesis**: Tightening `TOKAMAX_MAX_LOGIT_CONST` from v052's `30.0` to
`20.0` may further reduce forward Splash residual time while preserving the
normal loss trajectory.

**Mechanism**: Keep v052 fixed and change only `TOKAMAX_MAX_LOGIT_CONST=20.0`.

**Predicted signal**: Loss stays near the v052 trajectory, ending around
12.05, and throughput exceeds v052's 50,376 tok/s / 36.1% MFU.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v052's 50,376 tok/s / 36.1% MFU.

## Setup

GKE workload `alekseyv-qwen3-v053`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v053-shard-acts-s8k-bs3-tokamax-splash-maxlogit20`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=20.0`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but not above v052:

- Worker 0: 50,375 tok/s total, 6,297 tok/s/chip, 36.1% MFU.
- Worker 1: 50,356 tok/s total, 6,295 tok/s/chip, 36.1% MFU.
- Step time stabilized around 3.90 s after warmup.
- Loss trajectory stayed normal, ending at 12.0511.

This is essentially tied with but slightly below v052's 50,376 tok/s / 36.1%
MFU. Keep `TOKAMAX_MAX_LOGIT_CONST=30.0` as the current bound.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v053-shard-acts-s8k-bs3-tokamax-splash-maxlogit20/2026_06_03_04_27_42`
- **MXU utilization**: 56.0%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 48.2%.
- Custom-call: 23.4%.
- Loop fusion: 18.5%.
- Data formatting: 4.0%.

The custom-call split is effectively identical to v052: DKV 12.3% and forward
residuals 11.1%.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v053-shard-acts-s8k-bs3-tokamax-splash-maxlogit20/hlo/`
- **Size**: 19.02 MiB
- **Object count**: 26

## Verdict

**Refuted.** Tightening the max-logit bound from 30.0 to 20.0 does not improve
the v052 frontier.
