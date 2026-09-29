---
title: "Qwen3 JAX v275 - Confirmed RS3 selective-resources frontier with scoped VMEM 100864"
type: experiment
hypothesis: "On the confirmed RS3+AR2 selective-resources frontier, raising scoped VMEM from 100352 KiB to 100864 KiB may give the scheduler enough VMEM to recover a faster train-step schedule while staying within the new HBM margin."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs3-ar2-selective-resources-vmem100864
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, latency, rs3, ar2, vmem, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v275 - Confirmed RS3 selective-resources frontier with scoped VMEM 100864

## Hypothesis under test

**Hypothesis**: v273 confirmed RS3+AR2 selective resources as the safer
frontier with **60,346 tok/s / 43.3% MFU**, **4358.1 ms / 68.2% MXU**, and
**0.3181 GiB** free HBM. Raising only
`--xla_tpu_scoped_vmem_limit_kib` from `100352` to `100864` may let the
scheduler choose a slightly better matmul/collective overlap schedule while
remaining inside the confirmed HBM margin.

**Mechanism**: Reuse v273 exactly, changing only scoped VMEM. Keep RS latency
multiplier `3`, AR latency multiplier `2`, selective resources forced on,
SparseCore reduce-scatter/all-reduce offload, SparseCore collective aggregator,
Tokamax Splash, MaxText CE, no-scan, remat, MLP up-proj-first image, and
activation sharding.

**Predicted signal**: Support requires clean finite loss and either throughput
above the v273/v272 band, or same-band throughput with XProf step time below
**4358.1 ms** and no material loss of free HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, throughput/profile regression, or no useful schedule/profile
change relative to v273.

## Setup

GKE workload: `alekseyv-q3-v275-vmem100864`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v275-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-vmem100864`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduler isolate only. Model code, params, math, fake-data setup,
image, mesh, batch, latency multipliers, selective resources, SparseCore
offload stack, and all trainer flags match v273 except scoped VMEM is raised
from `100352` to `100864` KiB.

## Results

Workload `alekseyv-q3-v275-vmem100864` completed cleanly on both slices with
normal loss ending at **12.0459** and `EXIT_CODE=0`.

| Worker | Avg throughput, steps 2-19 | Tok/s/chip | Approx MFU | Best observed post-profile step |
|--------|----------------------------|------------|------------|---------------------------------|
| 0 | **60,284 tok/s** | **7,536** | **43.2%** | 60,423 tok/s |
| 1 | **60,311 tok/s** | **7,539** | **43.2%** | 60,423 tok/s |

This is below v273's confirmed worker0 **60,346 tok/s / 43.3% MFU** and below
v272's provisional best worker average **60,351 tok/s / 43.3% MFU**.

## Profile

- **Source**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v275-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-vmem100864`
- **xprof run**: `2026-06-05-qwen3-jax-v275-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-vmem100864/2026_06_05_17_22_51`
- **Average step time**: **4360.2 ms**
- **MXU utilization**: **68.1%**
- **Peak HBM**: **30.93 GiB / 31.25 GiB**
- **Free HBM at peak**: **0.3181 GiB**
- **Stack reservation / heap allocation**: **23.0797 GiB / 7.8483 GiB**
- **Idle**: **0.3%** of op-profile total time

Top op-profile buckets for `jit_train_step`:

| Bucket | Time | Share | Notes |
|--------|------|-------|-------|
| `convolution fusion` | 56,861.2 ms | 54.3% | Same aggregate matmul count as v273, slightly slower profile |
| `custom-call` | 29,402.8 ms | 28.1% | Splash attention same band |
| `loop fusion` | 10,728.4 ms | 10.2% | Higher than v273 |

## HLO Dump

- **Local copy**: `/tmp/qwen3-v275-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **SHA256**: `6e82cf40ff59806506583187b0d32c3242ae906e8aa6a6f49a2e725aa2c747a6`
- **Size**: **19,915,428 bytes / 119,347 lines**
- **v273 comparison**: not byte-identical (`cmp_v273=1`)
- **Counts**:
  - `all-gather`: 6997
  - `all-reduce`: 165
  - `reduce-scatter`: 2774
  - `async-start` / `async-done`: 394 / 394
  - `convolution`: 3081
  - `custom-call`: 3849
  - `fusion`: 30896
  - `copy`: 22111
  - `tokamax`: 2
  - `jvp_jit_silu__`: 576

## Verdict

**Refuted.** Raising scoped VMEM to `100864` selects a distinct HLO with the
same main collective/matmul/custom-call counts and six fewer `copy` text
matches, but it does not improve runtime. Trainer throughput drops to the
**43.2% MFU** rounded band and XProf worsens to **4360.2 ms** from v273's
**4358.1 ms**. Keep scoped VMEM `100352` on the carried RS3+AR2
selective-resources frontier.
