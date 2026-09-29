---
title: "Qwen3 JAX v260 - MLP up projection first with RS3+AR2 rerun"
type: experiment
hypothesis: "An exact rerun of v259 should reproduce the MLP up-proj-first + RS3+AR2 HLO and throughput band if the provisional 43.0% MFU result is real rather than run noise."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first
status: confirmed-same-band
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8, rerun]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v260 - MLP up projection first with RS3+AR2 rerun

## Hypothesis under test

**Hypothesis**: v259's **59,926 tok/s / 43.0% MFU**, best observed
**60,028 tok/s**, and **4390.3 ms / 67.7% MXU** XProf signal are reproducible
under an exact rerun.

**Mechanism**: Exact rerun of v259:

- Same image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- Same model/code: MLP computes `up_proj` before `gate_proj`
- Same runtime stack: RS latency multiplier `3`, AR latency multiplier `2`,
  Tokamax Splash, MaxText CE, no-scan, remat, activation sharding, SparseCore
  reduce-scatter/all-reduce offload plus aggregator

**Predicted signal**: Support requires clean finite loss, reproduced HLO hash
or clearly equivalent optimized HLO, and full-window throughput in the v259 band
(about **59.9k tok/s / 43.0% MFU**) with XProf near **4390 ms**.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, or a fall back to the v230/v236/v258 band without reproducing
v259's profile/throughput signal.

## Setup

GKE workload: `alekseyv-q3-v260-mlp-up-rs3-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v260-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime-only exact rerun of v259. Numerical equivalence is gated by the normal
loss trajectory; compile equivalence is gated by HLO hash and Tokamax markers.

## Results

- **Worker 0 summary**: 59,828 tok/s, 7,478 tok/s/chip, approx **42.9% MFU**.
- **Worker 1 summary**: 59,903 tok/s, 7,488 tok/s/chip, approx **43.0% MFU**.
- **Best observed post-profile step**: worker1 step 15 reached **60,030 tok/s**.
- **Loss**: normal fake-data trajectory, ending at **12.0460** on both workers.
- **Exit**: `EXIT_CODE=0`.

The full-window worker1 summary reproduces the rounded 43.0% band but not
v259's 59,926 tok/s average. Worker0 had one slow post-profile step 15
(58,322 tok/s) that depresses its full-window average.

## Profile

- **XProf run**:
  `2026-06-05-qwen3-jax-v260-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-rerun/2026_06_05_13_53_40`
- **Average step time**: **4389.7 ms**
- **MXU utilization**: **67.8%**
- **Peak HBM**: **30.94 GiB** of 31.25 GiB
- **Idle**: 0.3% in op profile
- **Top device buckets**:
  - `convolution fusion`: 57,135.2 ms, 54.2%
  - `custom-call`: 29,416.5 ms, 27.9%
  - `loop fusion`: 10,979.4 ms, 10.4%

## HLO Dump

- **Local optimized train-step HLO**:
  `/tmp/qwen3-v260-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **SHA256**:
  `57b4add6fe5afe1d02e0423086570ab999f02a2f6185bf935a208f7928d4b26a`
- **Size**: 19,580,278 bytes, 116,730 lines
- **v260 vs v259 optimized HLO**: byte-identical
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

## Verdict

**Confirmed same-band / carry schedule.**

v260 reproduced v259's optimized HLO byte-for-byte and improved the profiled
step-time signal from **4390.3 ms** to **4389.7 ms** with **67.8% MXU**. The
train-log throughput remains noisy around the same 42.9-43.0% nominal MFU band:
v259 remains the best full-window summary at **59,926 tok/s**, while v260
reproduces rounded **43.0% MFU** on worker1 and the **60,030 tok/s** best-step
signal. Carry the MLP up-proj-first + RS3+AR2 schedule as the best current
native-JAX frontier, with HLO/profile confirmation and small train-log margin.
