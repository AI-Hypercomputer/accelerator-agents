---
title: "Qwen3 JAX v266 - MLP up projection first with RS4 and default AR latency"
type: experiment
hypothesis: "The confirmed RS4+AR2 frontier changed the HLO family relative to the earlier RS3 AR-default test; removing only the all-reduce latency multiplier on the RS4 HLO may improve or simplify scheduling without losing the RS4 benefit."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs4-ar-default
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v266 - MLP up projection first with RS4 and default AR latency

## Hypothesis under test

**Hypothesis**: v262 showed that default all-reduce latency was worse on the
RS3 MLP up-proj-first HLO, but v263/v264 moved the frontier to a distinct
RS4+AR2 HLO. This tests whether AR latency multiplier `2` is still required
after RS4 changed the schedule, or whether the all-reduce delay is now
over-tuned.

**Mechanism**: Reuse the v264 runtime stack and image, keep
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=4`, and remove only
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`. Keep Tokamax Splash,
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

GKE workload: `alekseyv-q3-v266-mlp-up-rs4-ar1`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v266-no-scan-sparsecore-rs-ar-rs4-mlp-up-proj-first-ar-default`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduling-flag isolate only. Model code, params, math, random fake-data
setup, and MLP up-proj-first image match v263-v265.

## Results

Completed cleanly on `alekseyv-q3-v266-mlp-up-rs4-ar1` with finite loss and
`EXIT_CODE=0`. Removing AR latency multiplier `2` on the RS4 HLO did not
improve throughput.

- Worker0 measured steps 2-19: **59,942 tok/s / 7,493 tok/s/chip / 43.0% MFU**
- Worker1 measured steps 2-19: **59,917 tok/s / 7,490 tok/s/chip / 43.0% MFU**
- Best observed single step: worker1 step 15, **60,045 tok/s**
- Profile-window steps: worker0 **59,864 / 59,848 / 59,849 tok/s** for steps
  12-14; worker1 had one slower trace-start step at **59,427 tok/s**

## Profile

Primary XProf run:
`2026-06-05-qwen3-jax-v266-no-scan-sparsecore-rs-ar-rs4-mlp-up-proj-first-ar-default/2026_06_05_15_19_00`.

- Step time: **4388.5 ms**
- MXU utilization: **67.8%**
- Peak HBM: **30.93 GiB** / 31.25 GiB, **0.32 GiB free**
- Op profile: `jit_train_step` accounts for **99.7%** of time; idle is
  **0.3%**; convolution fusion and Splash custom-calls remain the dominant
  buckets.

The slight HBM reduction versus v264 does not compensate for worse step time
and lower train-log throughput.

## HLO Dump

Optimized train-step HLO:

- File: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256: `524d65de4503a2c4ae5aad00ebe173c6377e0f66256348db7e2a21a18a44b2c1`
- Size: **19,531,345 bytes / 116,454 lines**
- Counts: all-gather **6886**, all-reduce **165**, reduce-scatter **2774**,
  async-start/done **364/364**, convolution **3083**, custom-call **3539**,
  fusion **30583**, copy **21441**, `tokamax` **2**, `jvp_jit_silu__` **576**

Aggregate counts match v264, but the HLO is not byte-identical (`cmp_v264=1`).

## Verdict

**Refuted/tie.** Default AR latency on top of RS4 changes the HLO hash but not
the useful schedule shape, and it regresses XProf step time to **4388.5 ms**.
Keep **RS4+AR2** from v264 as the confirmed native-JAX frontier.
