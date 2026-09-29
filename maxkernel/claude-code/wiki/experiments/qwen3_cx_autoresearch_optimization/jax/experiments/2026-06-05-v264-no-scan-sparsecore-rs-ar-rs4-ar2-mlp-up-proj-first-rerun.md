---
title: "Qwen3 JAX v264 - MLP up projection first RS4+AR2 rerun"
type: experiment
hypothesis: "An exact rerun of v263 should reproduce the RS4+AR2 MLP up-proj-first HLO, 4387.5 ms XProf step time, and 43.0% MFU throughput band if the provisional high-water mark is real."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs4-ar2-rerun
status: confirmed-supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8, rerun]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v264 - MLP up projection first RS4+AR2 rerun

## Hypothesis under test

**Hypothesis**: v263's **59,947 tok/s / 43.0% MFU**, best observed
**60,045 tok/s**, and **4387.5 ms / 67.8% MXU** XProf signal are reproducible
under an exact rerun.

**Mechanism**: Exact rerun of v263:

- Same image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- Same model/code: MLP computes `up_proj` before `gate_proj`
- Same runtime stack: RS latency multiplier `4`, AR latency multiplier `2`,
  Tokamax Splash, MaxText CE, no-scan, remat, activation sharding, SparseCore
  reduce-scatter/all-reduce offload plus aggregator

**Predicted signal**: Support requires clean finite loss, reproduced HLO hash
or clearly equivalent optimized HLO, and full-window throughput in the v263 band
(about **59.9k tok/s / 43.0% MFU**) with XProf near or below **4388 ms**.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, failure to reproduce the v263 HLO family, or a fall back to the
v259/v260 RS3+AR2 band without reproducing v263's profile signal.

## Setup

GKE workload: `alekseyv-q3-v264-mlp-up-rs4-ar2-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v264-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime-only exact rerun of v263. Numerical equivalence is gated by the normal
loss trajectory; compile equivalence is gated by HLO hash and Tokamax markers.

## Results

Workload `alekseyv-q3-v264-mlp-up-rs4-ar2-r` completed cleanly on both workers
with normal loss ending at **12.0458** and `EXIT_CODE=0`.

- Worker0: **59,954 tok/s / 7,494 tok/s/chip / 43.0% MFU**
- Worker1: **59,933 tok/s / 7,492 tok/s/chip / 43.0% MFU**
- Best observed post-profile step: worker1 step 15 at **60,066 tok/s**
  (worker0 step 15 at **60,062 tok/s**)

The rerun reproduces v263's rounded **43.0% MFU** band and slightly improves
the best-worker full-window high-water mark from **59,947** to **59,954
tok/s**.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v264-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-rerun/2026_06_05_14_51_03`.

- Average step time: **4387.7 ms**
- MXU utilization: **67.9%**
- Peak HBM: **30.94 GiB** of **31.25 GiB**
- Idle: **85.402 ms / 0.2%** on the queried host
- Top op buckets on the queried host: convolution fusion **28,609.652 ms /
  54.4%**, custom-call **14,715.956 ms / 28.0%**, loop fusion
  **5,487.012 ms / 10.4%**

This reproduces v263's improved profile band (**4387.5 ms / 67.8% MXU**) and
stays ahead of the previous confirmed RS3+AR2 v260 profile (**4389.7 ms /
67.8% MXU**).

## HLO Dump

Optimized train-step HLO:
`/tmp/qwen3-v264-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- SHA256:
  `09b78e1542bfb74338809203b5a8cb1e7cc4a30c311e26432affa9b25ca1d9a3`
- Size: **19,531,349 bytes / 116,454 lines**
- `cmp` against v263 optimized HLO: byte-identical
- `cmp` against v260 optimized HLO: different
- Key counts: `all-gather` 6886, `all-reduce` 165, `reduce-scatter` 2774,
  `async-start` 364, `async-done` 364, `convolution` 3083, `custom-call`
  3539, `fusion` 30583, `copy` 21441, `tokamax` 2

The v263/v264 RS4+AR2 HLO family is now reproduced exactly.

## Verdict

**Confirmed supported.** Exact rerun reproduces the v263 HLO byte-for-byte,
keeps the same improved profile band, and raises the best full-window
throughput to **59,954 tok/s / 43.0% MFU**. Carry MLP up-proj-first + RS4+AR2
as the new confirmed native-JAX frontier, replacing the v259/v260 RS3+AR2
frontier.
