---
title: "Qwen3 JAX v259 - MLP up projection first with RS3+AR2"
type: experiment
hypothesis: "The v258 MLP up-proj-first source ordering may compose with the v236 RS3+AR2 latency-multiplier schedule to convert the small profiled step-time gain into a clearer throughput gain."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first
status: provisional-supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v259 - MLP up projection first with RS3+AR2

## Hypothesis under test

**Hypothesis**: v258 showed that computing `up_proj` before `gate_proj` changes
the optimized train-step HLO and improves the profiled step-time signal to
**4391.1 ms**, while holding the train-log throughput in the v230/v236 band.
v236 showed that RS latency multiplier `3` plus AR latency multiplier `2` is a
same-band confirmed alternative to v230's RS2+AR2. Combining the v258 source
ordering with v236's RS3+AR2 schedule may produce a clearer primary-metric gain
than either same-band change alone.

**Mechanism**: Reuse the v258 image
`<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
and change only the runtime flag
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier` from `2` to `3`.
Keep AR latency multiplier `2` and all other v258 flags/environment fixed.

**Predicted signal**: Support requires clean finite loss and throughput above
the current same-band high point (**59,904 tok/s / rounded 43.0% MFU**) or an
XProf profile materially below v258's **4391.1 ms** without train-log
regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax HLO markers, throughput below the v230/v236/v258 same-band, or no
profile-step improvement over v258.

## Setup

GKE workload: `alekseyv-q3-v259-mlp-up-rs3`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v259-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14
- **Runtime delta vs v258**: RS latency multiplier `3` instead of `2`

## Equivalence

Same image and model math as v258. The runtime-only delta should not affect
numerical semantics; finite loss and matching loss trajectory gate validity.

## Results

- **Worker 0 summary**: 59,896 tok/s, 7,487 tok/s/chip, approx **42.9% MFU**.
- **Worker 1 summary**: 59,926 tok/s, 7,491 tok/s/chip, approx **43.0% MFU**.
- **Best observed post-profile step**: worker1 step 15 reached **60,028 tok/s**.
- **Loss**: normal fake-data trajectory, ending at **12.0460** on both workers.
- **Exit**: `EXIT_CODE=0`.

This is the best train-log same-band point so far and modestly exceeds v258's
profiled step-time signal.

## Profile

- **XProf run**:
  `2026-06-05-qwen3-jax-v259-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first/2026_06_05_13_40_10`
- **Average step time**: **4390.3 ms**
- **MXU utilization**: **67.7%**
- **Peak HBM**: **30.94 GiB** of 31.25 GiB
- **Idle**: 0.3% in op profile
- **Top device buckets**:
  - `convolution fusion`: 57,135.1 ms, 54.2%
  - `custom-call`: 29,416.9 ms, 27.9% (mostly Splash attention)
  - `loop fusion`: 10,981.1 ms, 10.4%

## HLO Dump

- **Local optimized train-step HLO**:
  `/tmp/qwen3-v259-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **SHA256**:
  `57b4add6fe5afe1d02e0423086570ab999f02a2f6185bf935a208f7928d4b26a`
- **Size**: 19,580,278 bytes, 116,730 lines
- **Counts**:
  - `all-gather`: 6923
  - `all-reduce`: 165
  - `reduce-scatter`: 2774
  - `async-start`: 372
  - `async-done`: 372
  - `convolution`: 3073
  - `custom-call`: 3548
  - `fusion`: 30724
  - `copy`: 21325
  - `tokamax`: 2
  - `jvp_jit_silu__`: 576

The HLO is distinct from v258 and contains the expected Tokamax markers.

## Verdict

**Provisional supported; exact rerun required.**

The v258 source-order change composes slightly better with RS3+AR2 than with
RS2+AR2: v259 improves the XProf step time from **4391.1 ms** to **4390.3 ms**
and produces the best rounded train-log MFU point so far (**59,926 tok/s /
43.0% MFU**, best observed step **60,028 tok/s**). The margin is small, so do
not declare this confirmed until an exact rerun reproduces the same HLO and
throughput band.
