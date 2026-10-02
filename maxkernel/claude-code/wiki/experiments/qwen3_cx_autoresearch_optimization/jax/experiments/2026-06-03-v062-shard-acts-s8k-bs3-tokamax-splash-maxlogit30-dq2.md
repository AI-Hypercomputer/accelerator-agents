---
title: "Qwen3 JAX v062 - shard acts s8k bs3 tokamax splash maxlogit30 dq2"
type: experiment
hypothesis: "A less aggressive DQ reduction step count may reduce DKV time without the v054 wall-clock loss."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, dq-reduction, remat, maxtext-ce, activation-sharding, long-seq, invalid, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v062 - shard acts s8k bs3 tokamax splash maxlogit30 dq2

## Hypothesis under test

**Hypothesis**: `TOKAMAX_DQ_REDUCTION_STEPS=2` may recover some DKV reduction
from v054's `DQ_REDUCTION_STEPS=3` without paying the same wall-clock cost.

**Mechanism**: Keep the v056 frontier fixed (`TOKAMAX_MAX_LOGIT_CONST=30.0`,
target shape, scan/remat/Splash/MaxText CE/activation sharding, scheduler
rerun3) and set only `TOKAMAX_DQ_REDUCTION_STEPS=2`.

**Predicted signal**: Loss stays near the v052-v061 trajectory, ending around
12.05, and throughput exceeds v056's 50,383 tok/s / 36.1% MFU or reduces the
12.3% DKV bucket without the v054 wall-clock regression.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU
without a meaningful DKV reduction.

## Setup

GKE workload `alekseyv-qwen3-v062`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v062-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-dq2`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_DQ_REDUCTION_STEPS=2`
- **XLA flag**: keep v056's `--xla_latency_hiding_scheduler_rerun=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Invalid before train-step compile:

- Worker 0 exited with `ValueError: Invalid dq_reduction_steps: 2, only 3 or None are supported.`
- Worker 1 exited with the same error.
- No train steps completed.

## Profile

No profile was captured.

## HLO Dump

No useful train-step HLO was captured.

## Verdict

**Invalid.** Tokamax Splash only supports `dq_reduction_steps=None` or `3`, so
there is no DQ2 middle point to test.
