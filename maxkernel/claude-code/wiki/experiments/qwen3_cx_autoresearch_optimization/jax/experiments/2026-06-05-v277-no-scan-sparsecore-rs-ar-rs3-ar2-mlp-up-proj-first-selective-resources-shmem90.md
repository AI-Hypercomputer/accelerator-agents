---
title: "Qwen3 JAX v277 - Confirmed RS3 selective-resources frontier with shared-memory limit 90"
type: experiment
hypothesis: "On the confirmed RS3+AR2 selective-resources frontier, lowering the TPU scheduler shared-memory limit from 100 to 90 may force a different lower-memory schedule in the new HLO family; this is a high-side falsification after shmem95 proved byte-identical."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs3-ar2-selective-resources-shmem90
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, latency, rs3, ar2, shared-memory, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v277 - Confirmed RS3 selective-resources frontier with shared-memory limit 90

## Hypothesis under test

**Hypothesis**: v276 showed that shared-memory limit `95` leaves the v273 HLO
byte-identical. A stronger cap, `90`, may force a different scheduler solution
that improves overlap or HBM margin in the confirmed RS3+AR2 selective-resources
family. Prior lower-HBM schedules on older bases usually regressed throughput,
so this is primarily a quick falsification of the remaining shared-memory
branch.

**Mechanism**: Reuse v273 exactly, changing only
`--xla_tpu_scheduler_percent_shared_memory_limit=90`. Keep scoped VMEM
`100352`, RS latency multiplier `3`, AR latency multiplier `2`, selective
resources forced on, SparseCore reduce-scatter/all-reduce offload, SparseCore
collective aggregator, Tokamax Splash, MaxText CE, no-scan, remat, MLP
up-proj-first image, and activation sharding.

**Predicted signal**: Support requires clean finite loss and either throughput
above the v273/v272 band, or same-band throughput with XProf step time below
**4358.1 ms** and/or materially better HBM headroom.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, throughput/profile regression, or a byte-identical/no-gain
schedule.

## Setup

GKE workload: `alekseyv-q3-v277-shmem90`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v277-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem90`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduler isolate only. Model code, params, math, fake-data setup,
image, mesh, batch, scoped VMEM, latency multipliers, selective resources,
SparseCore offload stack, and all trainer flags match v273 except the
scheduler shared-memory limit is lowered from `100` to `90`.

## Results

Workload `alekseyv-q3-v277-shmem90` completed cleanly on both slices with
normal loss ending at **12.0459** and `EXIT_CODE=0`.

| Worker | Avg throughput, steps 2-19 | Tok/s/chip | Approx MFU | Best observed post-profile step |
|--------|----------------------------|------------|------------|---------------------------------|
| 0 | **60,294 tok/s** | **7,537** | **43.2%** | 60,393 tok/s |
| 1 | **60,269 tok/s** | **7,534** | **43.2%** | 60,396 tok/s |

This is below v273's confirmed **60,346 tok/s / 43.3% MFU** and below v276's
shmem95 tie.

## Profile

- **Source**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v277-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem90`
- **xprof run**: `2026-06-05-qwen3-jax-v277-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem90/2026_06_05_17_49_23`
- **Average step time**: **4361.9 ms**
- **MXU utilization**: **68.1%**
- **Peak HBM**: **29.03 GiB / 31.25 GiB**
- **Free HBM at peak**: **2.2202 GiB**
- **Stack reservation / heap allocation**: **23.0798 GiB / 5.9461 GiB**
- **Idle**: **0.3%** of op-profile total time

Top op-profile buckets for `jit_train_step`:

| Bucket | Time | Share | Notes |
|--------|------|-------|-------|
| `convolution fusion` | 56,919.9 ms | 54.4% | Higher than v273 despite byte-identical optimized text |
| `custom-call` | 29,406.8 ms | 28.1% | Splash attention slightly higher |
| `loop fusion` | 10,724.9 ms | 10.2% | Same band |

## HLO Dump

- **Local copy**: `/tmp/qwen3-v277-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
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

**Refuted.** Shared-memory limit `90` materially improves HBM headroom
(**2.2202 GiB** free versus v273's **0.3181 GiB**) but regresses the actual
performance target. Trainer throughput falls to the rounded **43.2% MFU** band
and XProf worsens to **4361.9 ms** from v273's **4358.1 ms**. The optimized
train-step text remains byte-identical to v273, so the gain is memory margin
only, not a better train-step schedule. Keep shared-memory limit `100`.
