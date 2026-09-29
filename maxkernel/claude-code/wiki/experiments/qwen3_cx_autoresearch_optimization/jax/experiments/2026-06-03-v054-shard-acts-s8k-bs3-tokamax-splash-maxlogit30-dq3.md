---
title: "Qwen3 JAX v054 - shard acts s8k bs3 tokamax splash maxlogit30 dq3"
type: experiment
hypothesis: "DQ reduction steps may improve the v052 max-logit frontier by reducing DKV time."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v054 - shard acts s8k bs3 tokamax splash maxlogit30 dq3

## Hypothesis under test

**Hypothesis**: Adding `TOKAMAX_DQ_REDUCTION_STEPS=3` to the v052
`TOKAMAX_MAX_LOGIT_CONST=30.0` frontier may reduce DKV time enough to improve
end-to-end throughput.

**Mechanism**: Keep v052 fixed and add only `TOKAMAX_DQ_REDUCTION_STEPS=3`.
v049 showed DQ reduction steps lower custom-call share but lose wall-clock
without max-logit; this run tests whether the combination is complementary.

**Predicted signal**: Loss stays near the v052 trajectory, ending around
12.05, and throughput exceeds v052's 50,376 tok/s / 36.1% MFU.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v052's 50,376 tok/s / 36.1% MFU.

## Setup

GKE workload `alekseyv-qwen3-v054`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v054-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-dq3`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_DQ_REDUCTION_STEPS=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but below v052:

- Worker 0: 50,163 tok/s total, 6,270 tok/s/chip, 36.0% MFU.
- Worker 1: 50,136 tok/s total, 6,267 tok/s/chip, 35.9% MFU.
- Step time stabilized around 3.92 s after warmup.
- Loss trajectory stayed normal, ending at 12.0510.

Compared with v052's 50,376 tok/s / 36.1% MFU, adding
`TOKAMAX_DQ_REDUCTION_STEPS=3` costs about 213-240 tok/s.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v054-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-dq3/2026_06_03_04_33_12`
- **MXU utilization**: 55.7%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 47.9%.
- Custom-call: 22.9%.
- Loop fusion: 18.6%.
- Data formatting: 4.0%.

`DQ_REDUCTION_STEPS=3` reduces DKV from v052's 12.3% to 11.9% and keeps
forward residuals around 11.0%, but the wall-clock result is still slower.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v054-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-dq3/hlo/`
- **Size**: 19.30 MiB
- **Object count**: 26

## Verdict

**Refuted.** DQ reduction steps remain a wall-clock loss even when combined
with the v052 max-logit bound.
