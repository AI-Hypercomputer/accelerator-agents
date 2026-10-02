---
title: "Qwen3 JAX v263 - MLP up projection first with RS4+AR2"
type: experiment
hypothesis: "The MLP up-proj-first HLO may tolerate a higher reduce-scatter latency multiplier than the pre-MLP-up schedule; testing RS4+AR2 closes the high-side RS bracket under the current frontier image."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs4-ar2
status: provisional-supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v263 - MLP up projection first with RS4+AR2

## Hypothesis under test

**Hypothesis**: v237 showed RS4+AR2 was worse before the MLP up-proj-first
source-order change, but v259/v260 selected a new HLO family and made RS3+AR2
the current frontier. Raising only the RS latency multiplier from `3` to `4`
tests whether the new MLP schedule shifts the high-side reduce-scatter
optimum.

**Mechanism**: Reuse the v260 runtime stack and image but change only
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3` to `4`. Keep AR
latency multiplier `2`, Tokamax Splash, MaxText CE, no-scan, remat, activation
sharding, SparseCore reduce-scatter and all-reduce offload, the SparseCore
collective aggregator, and all-gather on async collective fusion.

**Predicted signal**: Support requires clean finite loss and either throughput
above v259's best full-window **59,926 tok/s / 43.0% MFU**, or the same
throughput band with XProf step time below v260's **4389.7 ms** and unchanged
or lower HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, HBM pressure above the current **30.94 GiB** band with no
speedup, or a clean same/lower throughput/profile result.

## Setup

GKE workload: `alekseyv-q3-v263-mlp-up-rs4-ar2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v263-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduling-flag isolate only. Model code, params, math, random fake-data
setup, and MLP up-proj-first image match v259/v260.

## Results

Workload `alekseyv-q3-v263-mlp-up-rs4-ar2` completed cleanly on both workers
with normal loss ending at **12.0458** and `EXIT_CODE=0`.

- Worker0: **59,947 tok/s / 7,493 tok/s/chip / 43.0% MFU**
- Worker1: **59,926 tok/s / 7,491 tok/s/chip / 43.0% MFU**
- Best observed post-profile step: worker1 step 15 at **60,045 tok/s**
  (worker0 step 15 at **60,016 tok/s**)

Worker0's full-window average is a new native-JAX high-water mark, but the
margin over v259's **59,926 tok/s** is only **21 tok/s**. Treat this as
provisional until an exact rerun reproduces the HLO/profile/throughput band.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v263-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first/2026_06_05_14_37_45`.

- Average step time: **4387.5 ms**
- MXU utilization: **67.8%**
- Peak HBM: **30.94 GiB** of **31.25 GiB**
- Idle: **292.632 ms / 0.3%**
- Top op buckets: convolution fusion **57,135.666 ms / 54.2%**,
  custom-call **29,415.609 ms / 27.9%**, loop fusion
  **10,931.024 ms / 10.4%**

This is the best profiled step-time signal so far, improving v260's
**4389.7 ms** by **2.2 ms** while keeping peak HBM in the same band.

## HLO Dump

Optimized train-step HLO:
`/tmp/qwen3-v263-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- SHA256:
  `09b78e1542bfb74338809203b5a8cb1e7cc4a30c311e26432affa9b25ca1d9a3`
- Size: **19,531,349 bytes / 116,454 lines**
- `cmp` against v260/v262 optimized HLO: different
- Key counts: `all-gather` 6886, `all-reduce` 165, `reduce-scatter` 2774,
  `async-start` 364, `async-done` 364, `convolution` 3083, `custom-call`
  3539, `fusion` 30583, `copy` 21441, `tokamax` 2

Relative to v260, RS4+AR2 selects a smaller HLO with fewer all-gathers,
async-start/done events, custom-calls, and fusions, but more copies and
convolution text occurrences.

## Verdict

**Provisional supported.** The result is small but coherent: new best
full-window worker average (**59,947 tok/s**) and best XProf step time
(**4387.5 ms**) with normal loss, valid Tokamax markers, and unchanged peak HBM
band.

Do not declare RS4+AR2 confirmed yet. The margin is tiny, and v237's pre-MLP-up
RS4 result was a same-band regression. Exact rerun is required before carrying
RS4 over the v259/v260 confirmed RS3+AR2 frontier.
