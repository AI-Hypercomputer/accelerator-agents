---
title: "Qwen3 JAX v262 - MLP up projection first RS3 with default AR latency"
type: experiment
hypothesis: "The MLP up-proj-first HLO may have changed the all-reduce latency multiplier optimum; dropping AR latency multiplier 2 while keeping RS latency multiplier 3 tests whether AR2 is still required on the current frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-ar-default
status: completed-refuted-tie
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v262 - MLP up projection first RS3 with default AR latency

## Hypothesis under test

**Hypothesis**: v259/v260 confirmed MLP up-proj-first + RS3+AR2 as the current
native-JAX frontier, but the AR latency multiplier was inherited from the
pre-MLP-up schedule. Dropping only
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2` tests whether the new
HLO family still needs AR2 or whether the default all-reduce latency estimate
can pick an equal or better schedule.

**Mechanism**: Reuse the v260 runtime stack and image but remove only the AR
latency multiplier override. Keep RS latency multiplier `3`, Tokamax Splash,
MaxText CE, no-scan, remat, activation sharding, SparseCore reduce-scatter and
all-reduce offload, the SparseCore collective aggregator, and all-gather on
async collective fusion.

**Predicted signal**: Support requires clean finite loss and either throughput
above v259's best full-window **59,926 tok/s / 43.0% MFU**, or the same
throughput band with XProf step time below v260's **4389.7 ms** and unchanged
HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, changed HLO with lower throughput/profile, or byte-identical
HLO with no measurable throughput or profile gain.

## Setup

GKE workload: `alekseyv-q3-v262-mlp-up-rs3-ar1`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v262-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-ar-default`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduling-flag isolate only. Model code, params, math, random fake-data
setup, and MLP up-proj-first image match v259/v260/v261.

## Results

Workload `alekseyv-q3-v262-mlp-up-rs3-ar1` completed cleanly on both workers
with normal loss ending at **12.0460** and `EXIT_CODE=0`.

- Worker0: **59,918 tok/s / 7,490 tok/s/chip / 43.0% MFU**
- Worker1: **59,900 tok/s / 7,487 tok/s/chip / 42.9% MFU**
- Best observed post-profile step: worker0 step 15 at **60,024 tok/s**
  (worker1 step 15 at **60,021 tok/s**)

The best worker average is same-band with the v259/v260 frontier but does not
beat v259's **59,926 tok/s** best full-window result.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v262-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-ar-default/2026_06_05_14_24_01`.

- Average step time: **4391.2 ms**
- MXU utilization: **67.8%**
- Peak HBM: **30.93 GiB** of **31.25 GiB**
- Idle: **259.493 ms / 0.2%**
- Top op buckets: convolution fusion **57,136.647 ms / 54.2%**,
  custom-call **29,415.803 ms / 27.9%**, loop fusion
  **10,979.047 ms / 10.4%**

The profile is effectively tied with v260 but slightly slower on average step
time (**4391.2 ms** vs **4389.7 ms**), despite a small memory reduction.

## HLO Dump

Optimized train-step HLO:
`/tmp/qwen3-v262-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- SHA256:
  `22d309bd3f749edd48134a95832505670527871f796e8708cd47142b45d737de`
- Size: **19,580,274 bytes / 116,730 lines**
- `cmp` against v260 optimized HLO: different
- Key counts: `all-gather` 6923, `all-reduce` 165, `reduce-scatter` 2774,
  `async-start` 372, `async-done` 372, `convolution` 3073, `custom-call`
  3548, `fusion` 30724, `copy` 21325, `tokamax` 2

The text diff is localized to scheduling/typing of async all-gather fusions
around rematerialized MLP dot-general and RMSNorm/mul work. The aggregate HLO
surface and op counts remain the same as v260.

## Verdict

**Refuted/tie.** Removing AR latency multiplier `2` produces a valid distinct
schedule but does not improve the frontier. Throughput stays in the same
rounded **43.0% MFU** band, the best full-window average remains below v259,
and XProf step time is slightly worse than v260.

Keep `--xla_tpu_sparse_core_all_reduce_latency_multiplier=2` in the carried
MLP up-proj-first + RS3+AR2 schedule.
