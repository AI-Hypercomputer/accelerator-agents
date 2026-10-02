---
title: "Qwen3 JAX v276 - Confirmed RS3 selective-resources frontier with shared-memory limit 95"
type: experiment
hypothesis: "On the confirmed RS3+AR2 selective-resources frontier, lowering the TPU scheduler shared-memory limit from 100 to 95 may select a better overlap schedule in the new HLO family despite earlier RS2-era refutations."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs3-ar2-selective-resources-shmem95
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, latency, rs3, ar2, shared-memory, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v276 - Confirmed RS3 selective-resources frontier with shared-memory limit 95

## Hypothesis under test

**Hypothesis**: v273's RS3+AR2 selective-resources HLO is a different family
from the earlier RS2-era shmem95 refutation. Lowering
`--xla_tpu_scheduler_percent_shared_memory_limit` from `100` to `95` may alter
placement/overlap enough to improve the **4358.1 ms / 43.3% MFU** v273
profile while staying inside the **0.3181 GiB** free-HBM margin.

**Mechanism**: Reuse v273 exactly, changing only
`--xla_tpu_scheduler_percent_shared_memory_limit=95`. Keep scoped VMEM
`100352`, RS latency multiplier `3`, AR latency multiplier `2`, selective
resources forced on, SparseCore reduce-scatter/all-reduce offload, SparseCore
collective aggregator, Tokamax Splash, MaxText CE, no-scan, remat, MLP
up-proj-first image, and activation sharding.

**Predicted signal**: Support requires clean finite loss and either throughput
above the v273/v272 band, or same-band throughput with XProf step time below
**4358.1 ms** and no material loss of free HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, throughput/profile regression, or a byte-identical/no-gain
schedule.

## Setup

GKE workload: `alekseyv-q3-v276-shmem95`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v276-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem95`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduler isolate only. Model code, params, math, fake-data setup,
image, mesh, batch, scoped VMEM, latency multipliers, selective resources,
SparseCore offload stack, and all trainer flags match v273 except the
scheduler shared-memory limit is lowered from `100` to `95`.

## Results

Workload `alekseyv-q3-v276-shmem95` completed cleanly on both slices with
normal loss ending at **12.0459** and `EXIT_CODE=0`.

| Worker | Avg throughput, steps 2-19 | Tok/s/chip | Approx MFU | Best observed post-profile step |
|--------|----------------------------|------------|------------|---------------------------------|
| 0 | **60,344 tok/s** | **7,543** | **43.3%** | 60,440 tok/s |
| 1 | **60,325 tok/s** | **7,541** | **43.3%** | 60,466 tok/s |

This is same-band but not better than v273's confirmed worker0 **60,346 tok/s**
or v272's provisional worker1 **60,351 tok/s**.

## Profile

- **Source**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v276-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem95`
- **xprof run**: `2026-06-05-qwen3-jax-v276-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem95/2026_06_05_17_36_25`
- **Average step time**: **4358.5 ms**
- **MXU utilization**: **68.2%**
- **Peak HBM**: **30.93 GiB / 31.25 GiB**
- **Free HBM at peak**: **0.3181 GiB**
- **Stack reservation / heap allocation**: **23.0798 GiB / 7.8482 GiB**
- **Idle**: **0.2%** of op-profile total time

Top op-profile buckets for `jit_train_step`:

| Bucket | Time | Share | Notes |
|--------|------|-------|-------|
| `convolution fusion` | 56,840.3 ms | 54.3% | Same HLO family as v273 |
| `custom-call` | 29,396.6 ms | 28.1% | Splash attention same band |
| `loop fusion` | 10,722.8 ms | 10.3% | Same band as v273 |

## HLO Dump

- **Local copy**: `/tmp/qwen3-v276-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **SHA256**: `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
- **Size**: **19,915,660 bytes / 119,349 lines**
- **v273 comparison**: byte-identical (`cmp_v273=0`)
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

**Refuted/tie.** Lowering scheduler shared-memory limit to `95` produces a
byte-identical optimized train-step HLO to v273 and preserves the same HBM
profile, but it does not improve the measured result. The best worker average
is **60,344 tok/s**, just below v273's **60,346 tok/s**, and XProf is
**4358.5 ms** versus v273's **4358.1 ms**. Keep
`--xla_tpu_scheduler_percent_shared_memory_limit=100` on the carried v273
frontier.
