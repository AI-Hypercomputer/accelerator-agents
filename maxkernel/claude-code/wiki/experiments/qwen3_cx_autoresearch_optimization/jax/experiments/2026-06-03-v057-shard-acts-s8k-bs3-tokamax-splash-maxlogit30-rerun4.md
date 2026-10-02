---
title: "Qwen3 JAX v057 - shard acts s8k bs3 tokamax splash maxlogit30 rerun4"
type: experiment
hypothesis: "Another XLA latency-hiding scheduler rerun may continue the marginal v056 throughput gain."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v057 - shard acts s8k bs3 tokamax splash maxlogit30 rerun4

## Hypothesis under test

**Hypothesis**: Increasing `--xla_latency_hiding_scheduler_rerun` from 3 to 4
may continue the marginal scheduler gain observed in v056.

**Mechanism**: Keep v056 fixed (`TOKAMAX_MAX_LOGIT_CONST=30.0`, target shape,
scan/remat/Splash/MaxText CE/activation sharding) and change only the XLA
latency-hiding scheduler rerun count in `LIBTPU_INIT_ARGS`.

**Predicted signal**: Loss stays near the v052/v056 trajectory, ending around
12.05, and throughput exceeds v056's 50,383 tok/s / 36.1% MFU.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU.

## Setup

GKE workload `alekseyv-qwen3-v057`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v057-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-rerun4`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: `--xla_latency_hiding_scheduler_rerun=4` instead of v056's `3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but below v056:

- Worker 0: 50,369 tok/s total, 6,296 tok/s/chip, 36.1% MFU.
- Worker 1: 50,354 tok/s total, 6,294 tok/s/chip, 36.1% MFU.
- Step time stabilized around 3.90 s after warmup.
- Loss trajectory stayed normal, ending at 12.0508.

Compared with v056's 50,383 tok/s / 36.1% MFU, rerun4 gives back the tiny
rerun3 gain. Treat the scheduler-rerun axis as saturated at this shape.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v057-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-rerun4/2026_06_03_20_29_04`
- **MXU utilization**: 55.9%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.2%.

Top device buckets:

- Convolution fusion: 48.2%.
- Custom-call: 23.4%.
- Loop fusion: 18.5%.
- Data formatting: 4.0%.

The custom-call split is unchanged: DKV backward is about 12.3%, and forward
residuals are about 11.1%.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v057-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-rerun4/hlo/`
- **Size**: 31.82 MiB
- **Object count**: 1619

## Verdict

**Refuted.** `--xla_latency_hiding_scheduler_rerun=4` is clean and valid but
does not beat v056. Stop increasing scheduler reruns and pivot back to
attention custom-call scheduling/shape work.
