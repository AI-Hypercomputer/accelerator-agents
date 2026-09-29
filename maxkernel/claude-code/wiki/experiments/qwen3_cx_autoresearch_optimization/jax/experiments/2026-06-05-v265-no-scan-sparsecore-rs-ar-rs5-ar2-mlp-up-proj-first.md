---
title: "Qwen3 JAX v265 - MLP up projection first with RS5+AR2"
type: experiment
hypothesis: "The confirmed v264 RS4+AR2 frontier may not be the high-side reduce-scatter latency optimum; raising only RS latency multiplier to 5 tests whether the new MLP up-proj-first HLO benefits from even later reduce-scatter scheduling."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs5-ar2
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v265 - MLP up projection first with RS5+AR2

## Hypothesis under test

**Hypothesis**: v263/v264 showed that the MLP up-proj-first HLO shifted the
high-side reduce-scatter latency bracket from the earlier RS3/RS4 result:
RS4+AR2 selected a smaller HLO and improved the confirmed frontier to
**59,954 tok/s / 43.0% MFU** with **4387.7 ms** XProf step time. Raising only
the RS latency multiplier from `4` to `5` tests whether the high-side optimum is
still open or whether RS4 is the local peak.

**Mechanism**: Reuse the v264 runtime stack and image but change only
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=4` to `5`. Keep AR
latency multiplier `2`, Tokamax Splash, MaxText CE, no-scan, remat, activation
sharding, SparseCore reduce-scatter and all-reduce offload, the SparseCore
collective aggregator, and all-gather on async collective fusion.

**Predicted signal**: Support requires clean finite loss and either throughput
above v264's best full-window **59,954 tok/s / 43.0% MFU**, or the same
throughput band with XProf step time below v263/v264's **4387.5-4387.7 ms**
band and unchanged or lower HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, HBM pressure above the current **30.94 GiB** band with no
speedup, or a clean same/lower throughput/profile result.

## Setup

GKE workload: `alekseyv-q3-v265-mlp-up-rs5-ar2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v265-no-scan-sparsecore-rs-ar-rs5-ar2-mlp-up-proj-first`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduling-flag isolate only. Model code, params, math, random fake-data
setup, and MLP up-proj-first image match v263/v264.

## Results

Completed cleanly on `alekseyv-q3-v265-mlp-up-rs5-ar2` with finite loss and
`EXIT_CODE=0`. The RS5 HLO is distinct from v264, but the throughput result is
same-band and not better than the confirmed RS4+AR2 frontier.

- Worker0 measured steps 2-19: **59,955 tok/s / 7,494 tok/s/chip / 43.0% MFU**
- Worker1 measured steps 2-19: **59,927 tok/s / 7,491 tok/s/chip / 43.0% MFU**
- Best observed single step: worker0 step 15, **60,061 tok/s**
- Profile-window steps: worker0 **59,888 / 59,866 / 59,858 tok/s** for steps
  12-14; worker1 had one slower trace-start step at **59,408 tok/s**

## Profile

Primary XProf run:
`2026-06-05-qwen3-jax-v265-no-scan-sparsecore-rs-ar-rs5-ar2-mlp-up-proj-first/2026_06_05_15_05_08`.

- Step time: **4387.6 ms**
- MXU utilization: **67.9%**
- Peak HBM: **31.0 GiB** / 31.25 GiB, **0.24 GiB free**
- Op profile: `jit_train_step` accounts for **99.8%** of time; idle is
  **0.2%**; convolution fusion remains the dominant bucket.

This ties the v263/v264 step-time band (**4387.5-4387.7 ms**) but uses slightly
more memory than v264's **30.94 GiB** peak.

## HLO Dump

Optimized train-step HLO:

- File: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256: `f5cb5c9c74e10596bddc5a6247e2d58cb162dfc7e2147fe90615cb03b97bdee5`
- Size: **19,519,075 bytes / 116,419 lines**
- Counts: all-gather **6874**, all-reduce **165**, reduce-scatter **2774**,
  async-start/done **379/379**, convolution **3085**, custom-call **3554**,
  fusion **30532**, copy **21224**, `tokamax` **2**, `jvp_jit_silu__` **576**

The HLO is not byte-identical to v264 (`cmp_v264=1`): RS5 reduces all-gather
and copy counts but increases async-start/done and custom-call counts.

## Verdict

**Refuted/tie.** RS5+AR2 changes the compiled schedule but does not improve
throughput or XProf step time, and it raises peak HBM slightly. Keep
**RS4+AR2** from v264 as the confirmed native-JAX frontier.
