---
title: "Qwen3 JAX v261 - MLP up projection first RS3+AR2 without offload tracing"
type: experiment
hypothesis: "The confirmed v260 MLP up-proj-first + RS3+AR2 frontier may retain the same optimized HLO while reducing runtime/profile perturbation if active SparseCore reduce-scatter/all-reduce offload tracing is disabled."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-notrace
status: completed-refuted-tie
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, offload-tracing, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v261 - MLP up projection first RS3+AR2 without offload tracing

## Hypothesis under test

**Hypothesis**: v260 confirmed the MLP up-proj-first + RS3+AR2 schedule with a
byte-identical v259/v260 optimized HLO, **4389.7 ms** XProf step time, and a
rounded **43.0% MFU** throughput band. Earlier v238/v247 showed that disabling
active SparseCore offload tracing reduced profile volume but did not improve
the pre-MLP-up schedules. Retesting on the current frontier checks whether the
new HLO family is sensitive to this diagnostic instrumentation.

**Mechanism**: Reuse the v260 runtime stack and image but change only:

- `--xla_tpu_enable_reduce_scatter_offload_tracing=true` to `false`
- `--xla_tpu_enable_all_reduce_offload_tracing=true` to `false`

All-gather offload tracing remains `false` as in v260 because all-gather
SparseCore offload is disabled in the RS/AR split.

**Predicted signal**: Support requires clean finite loss and either throughput
above v259's best full-window **59,926 tok/s / 43.0% MFU**, or the same
throughput band with XProf step time below v260's **4389.7 ms** and unchanged
HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, changed HLO with worse profile, or a clean same/lower
throughput and profile tie.

## Setup

GKE workload: `alekseyv-q3-v261-mlp-up-rs3-notr`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v261-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-no-offload-tracing`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime tracing-flag isolate only. Model code, params, math, and random fake-data
setup match v259/v260.

## Results

Workload `alekseyv-q3-v261-mlp-up-rs3-notr` completed cleanly on both workers
with normal loss ending at **12.0460** and `EXIT_CODE=0`.

- Worker0: **59,901 tok/s / 7,488 tok/s/chip / 42.9% MFU**
- Worker1: **59,875 tok/s / 7,484 tok/s/chip / 42.9% MFU**
- Best observed post-profile step: **59,980 tok/s** on both workers

## Profile

XProf run:
`2026-06-05-qwen3-jax-v261-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-no-offload-tracing/2026_06_05_14_09_16`.

- Average step time: **4390.8 ms**
- MXU utilization: **67.8%**
- Peak HBM: **30.94 GiB** of **31.25 GiB**
- Idle: **298.827 ms / 0.3%**
- Top op buckets: convolution fusion **57,128.948 ms / 54.2%**,
  custom-call **29,416.517 ms / 27.9%**, loop fusion
  **10,978.019 ms / 10.4%**

Disabling active reduce-scatter/all-reduce offload tracing reduced profile
artifact volume from v260's **1,883,729,274 bytes** to **295,699,891 bytes**
(about **6.4x** smaller), but did not improve the performance counters.

## HLO Dump

Optimized train-step HLO:
`/tmp/qwen3-v261-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- SHA256:
  `57b4add6fe5afe1d02e0423086570ab999f02a2f6185bf935a208f7928d4b26a`
- Size: **19,580,278 bytes / 116,730 lines**
- `cmp` against v260 optimized HLO: byte-identical
- Key counts: `all-gather` 6923, `all-reduce` 165, `reduce-scatter` 2774,
  `async-start` 372, `async-done` 372, `convolution` 3073, `custom-call`
  3548, `fusion` 30724, `copy` 21325, `tokamax` 2

## Verdict

**Refuted/tie.** This is a useful diagnostics-volume flag but not a performance
lever for the current native-JAX frontier. The optimized HLO is byte-identical
to v259/v260, profile size is much smaller, and throughput remains in the same
rounded **42.9-43.0% MFU** band. However, v261 does not beat v259's
**59,926 tok/s** full-window result or v260's **4389.7 ms** XProf step time.

Do not carry no-offload-tracing for performance runs. It is acceptable for
smaller diagnostic profiles when detailed SparseCore offload tracing is not
needed.
