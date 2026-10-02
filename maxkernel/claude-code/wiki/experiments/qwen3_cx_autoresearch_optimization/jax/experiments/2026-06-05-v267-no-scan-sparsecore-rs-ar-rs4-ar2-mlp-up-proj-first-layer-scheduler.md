---
title: "Qwen3 JAX v267 - MLP up projection first with RS4+AR2 and layer scheduler"
type: experiment
hypothesis: "The confirmed RS4+AR2 HLO may benefit from explicitly pairing the latency-hiding layer scheduler with the SparseCore collective aggregator; earlier aggregator-era tests used this pair before the current MLP up-proj-first and RS4+AR2 schedule existed."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs4-ar2-layer-scheduler
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, layer-scheduler, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v267 - MLP up projection first with RS4+AR2 and layer scheduler

## Hypothesis under test

**Hypothesis**: v265/v266 bracketed the immediate RS/AR latency multipliers
around the confirmed v264 RS4+AR2 frontier. The next scheduler axis is to
explicitly enable the latency-hiding layer scheduler while keeping the
SparseCore collective aggregator. Earlier in the lane, this pair was the valid
way to use the layer scheduler; the current MLP up-proj-first + RS4+AR2 HLO has
not been checked with that pair.

**Mechanism**: Reuse the v264 runtime stack and image, keep RS latency
multiplier `4`, AR latency multiplier `2`, and add only
`--xla_tpu_enable_latency_hiding_layer_scheduler=true`. Keep Tokamax Splash,
MaxText CE, no-scan, remat, activation sharding, SparseCore reduce-scatter and
all-reduce offload, the SparseCore collective aggregator, and all-gather on
async collective fusion.

**Predicted signal**: Support requires clean finite loss and either throughput
above v264's best full-window **59,954 tok/s / 43.0% MFU**, or the same
throughput band with XProf step time below v264's **4387.7 ms** and unchanged
or lower HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, HBM pressure above the current **30.94 GiB** band with no
speedup, or a clean same/lower throughput/profile result.

## Setup

GKE workload: `alekseyv-q3-v267-mlp-up-rs4-ar2-ls`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v267-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-layer-scheduler`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduler-flag isolate only. Model code, params, math, random fake-data
setup, and MLP up-proj-first image match v263-v266.

## Results

Both workers failed during train-step compile before step 0 with
`RESOURCE_EXHAUSTED`; no train throughput or XProf profile was produced.

Worker0 error:

- `Used 66.54G of 31.25G hbm`
- `Exceeded hbm capacity by 35.29G`
- Total HBM usage at least **66.79 GiB**
- HLO temp: **66.46 GiB**
- Top allocations are repeated **768.00 MiB** rematerialized
  `bf16[4,8192,12288]` dot-general temps around checkpoint/remat MLP work.
- `EXIT_CODE=1`

## Profile

No profile: the program failed at compile before step 0.

## HLO Dump

Optimized train-step HLO was dumped despite compile failure:

- File: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256: `02275c00be2740bf453d7fec2fdae2dfac9181c67b3f43e86be518a16c547bfa`
- Size: **16,418,809 bytes / 98,751 lines**
- Counts: all-gather **4855**, all-reduce **165**, reduce-scatter **2774**,
  async-start/done **256/256**, convolution **3633**, custom-call **2211**,
  fusion **22832**, copy **27180**, `tokamax` **2**, `jvp_jit_silu__` **712**

The HLO is not byte-identical to v264 (`cmp_v264=1`). It is textually smaller
and has fewer collective/custom-call/fusion entries, but it schedules/remats
large MLP dot temps in a way that requires impossible HBM.

## Verdict

**Invalid/refuted.** Explicit latency-hiding layer scheduler on top of the
current SparseCore aggregator + RS4+AR2 stack is not viable: it changes the HLO
shape and blows train-step compile memory up to **66.54 GiB**. Keep the layer
scheduler disabled for the current frontier.
