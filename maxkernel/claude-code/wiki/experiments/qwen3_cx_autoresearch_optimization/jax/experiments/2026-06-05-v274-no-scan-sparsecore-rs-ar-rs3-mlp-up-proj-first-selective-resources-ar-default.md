---
title: "Qwen3 JAX v274 - Confirmed RS3 selective-resources frontier with AR default"
type: experiment
hypothesis: "On the newly confirmed RS3+AR2 selective-resources frontier, removing only the SparseCore all-reduce latency multiplier may improve or simplify the schedule without losing the confirmed HBM headroom."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs3-ar-default-selective-resources
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, latency, rs3, ar-default, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v274 - Confirmed RS3 selective-resources frontier with AR default

## Hypothesis under test

**Hypothesis**: v273 confirms RS3+AR2 selective resources as the safer
frontier (**60,346 tok/s / 43.3% MFU**, **4358.1 ms / 68.2% MXU**, **0.3181
GiB** free HBM). Returning only
`--xla_tpu_sparse_core_all_reduce_latency_multiplier` to its default may reduce
scheduler distortion or simplify the HLO while preserving the HBM margin.

**Mechanism**: Reuse v273 exactly, changing only the AR latency multiplier:
omit `--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`. Keep RS latency
multiplier `3`, selective resources forced on, scoped VMEM `100352`,
SparseCore reduce-scatter/all-reduce offload, SparseCore collective aggregator,
Tokamax Splash, MaxText CE, no-scan, remat, MLP up-proj-first image, and
activation sharding.

**Predicted signal**: Support requires clean finite loss and either throughput
above the v273/v272 band, or same-band throughput with XProf step time at or
below **4358.1 ms** and no loss of the **0.3181 GiB** free-HBM margin.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, throughput/profile regression toward v262/v266 AR-default
results, or materially worse HBM headroom.

## Setup

GKE workload: `alekseyv-q3-v274-sr-rs3-ard`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v274-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-selective-resources-ar-default`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduler isolate only. Model code, params, math, fake-data setup,
image, mesh, batch, RS latency multiplier, selective resources, scoped VMEM,
and all trainer flags match v273 except the AR latency multiplier is omitted.

## Results

Workload `alekseyv-q3-v274-sr-rs3-ard` completed cleanly on both slices with
normal loss ending at **12.0459** and `EXIT_CODE=0`.

| Worker | Avg throughput, steps 2-19 | Tok/s/chip | Approx MFU | Best observed post-profile step |
|--------|----------------------------|------------|------------|---------------------------------|
| 0 | **60,334 tok/s** | **7,542** | **43.3%** | 60,444 tok/s |
| 1 | **60,315 tok/s** | **7,539** | **43.2%** | 60,446 tok/s |

This is same-band but below v273's confirmed worker0 **60,346 tok/s** and below
v272's provisional **60,351 tok/s**. It also drops worker1 below the rounded
43.3% band in the trainer summary.

## Profile

- **Source**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v274-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-selective-resources-ar-default`
- **xprof run**: `2026-06-05-qwen3-jax-v274-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-selective-resources-ar-default/2026_06_05_17_07_37`
- **Average step time**: **4359.4 ms**
- **MXU utilization**: **68.1%**
- **Peak HBM**: **30.93 GiB / 31.25 GiB**
- **Free HBM at peak**: **0.3181 GiB**
- **Stack reservation / heap allocation**: **23.0798 GiB / 7.8482 GiB**
- **Idle**: **0.3%** of op-profile total time

Top op-profile buckets for `jit_train_step`:

| Bucket | Time | Share | Notes |
|--------|------|-------|-------|
| `convolution fusion` | 56,866.8 ms | 54.3% | Slightly higher than v273; train-step program changed |
| `custom-call` | 29,395.7 ms | 28.1% | Splash attention same band |
| `loop fusion` | 10,718.6 ms | 10.2% | Same band as v273 |

## HLO Dump

- **Local copy**: `/tmp/qwen3-v274-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **SHA256**: `361b10fd3f580e2829ae2a660ad81c4cf38bc62bfbcf0da033214495b2a5c44e`
- **Size**: **19,915,660 bytes / 119,349 lines**
- **v273 comparison**: not byte-identical (`cmp_v273=1`)
- **v269 comparison**: not byte-identical (`cmp_v269=1`)
- **Counts**:
  - `all-gather`: 6997
  - `all-reduce`: 165
  - `reduce-scatter`: 2774
  - `async-start` / `async-done`: 394 / 394
  - `convolution`: 3081
  - `custom-call`: 3849
  - `fusion`: 30896
  - `copy`: 22117
  - `tokamax`: 2
  - `jvp_jit_silu__`: 576

## Verdict

**Refuted.** AR default selects a distinct HLO with the same aggregate counts
and preserves the v273 HBM headroom, but it regresses the actual performance
signal: trainer averages are lower and XProf step time worsens to **4359.4 ms**
from v273's **4358.1 ms**. Keep
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2` on the carried RS3
selective-resources frontier.
