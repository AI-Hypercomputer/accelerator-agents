---
title: "Qwen3 JAX v273 - Selective-resources RS3+AR2 exact rerun"
type: experiment
hypothesis: "v272 preserved the v269 throughput band while improving free HBM from 0.0896 GiB to 0.3181 GiB; an exact rerun can determine whether RS3+AR2 selective resources is a real safer frontier or one-run scheduling noise."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs3-ar2-selective-resources-rerun
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, latency, rs3, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8, rerun]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v273 - Selective-resources RS3+AR2 exact rerun

## Hypothesis under test

**Hypothesis**: v272's RS3+AR2 selective-resources schedule is a reproducible
memory-headroom tie: it should preserve the **60.3k tok/s / 43.3% MFU** band
while keeping peak HBM near **30.93 GiB** rather than v269's **31.16 GiB**.

**Mechanism**: Exact rerun of v272. Model code, image, mesh, batch, profile
window, Splash/Tokamax/MaxText CE settings, scoped VMEM, SparseCore RS/AR
offload, SparseCore aggregator, selective resources, RS latency multiplier
`3`, and AR latency multiplier `2` all match v272.

**Predicted signal**: Support requires clean finite loss, optimized HLO
byte-identical to v272 or semantically matching the same RS3+AR2 family, XProf
step time near **4358 ms**, and peak free HBM materially above v269's
**0.0896 GiB** without throughput dropping below the v269/v272 band.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, throughput/profile regression toward the old pre-selective
RS3 band, or loss of the HBM headroom seen in v272.

## Setup

GKE workload: `alekseyv-q3-v273-sr-rs3-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Exact runtime rerun of v272. No model-code, math, fake-data, topology, image,
or flag changes are intended except the workload and GCS output names.

## Results

Workload `alekseyv-q3-v273-sr-rs3-r` completed cleanly on both slices with
normal loss ending at **12.0459** and `EXIT_CODE=0`.

| Worker | Avg throughput, steps 2-19 | Tok/s/chip | Approx MFU | Best observed post-profile step |
|--------|----------------------------|------------|------------|---------------------------------|
| 0 | **60,346 tok/s** | **7,543** | **43.3%** | 60,445 tok/s |
| 1 | **60,322 tok/s** | **7,540** | **43.3%** | 60,440 tok/s |

This confirms v272's same-band throughput result. The exact rerun is slightly
below v272's best worker average (**60,351 tok/s**) but still above v269's
confirmed best worker average (**60,338 tok/s**) by noise-level margin and
matches the same **43.3%** MFU band.

## Profile

- **Source**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun`
- **xprof run**: `2026-06-05-qwen3-jax-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun/2026_06_05_16_54_18`
- **Average step time**: **4358.1 ms**
- **MXU utilization**: **68.2%**
- **Peak HBM**: **30.93 GiB / 31.25 GiB**
- **Free HBM at peak**: **0.3181 GiB**
- **Stack reservation / heap allocation**: **23.0798 GiB / 7.8482 GiB**
- **Idle**: **0.3%** of op-profile total time

Top op-profile buckets for `jit_train_step`:

| Bucket | Time | Share | Notes |
|--------|------|-------|-------|
| `convolution fusion` | 56,834.9 ms | 54.3% | Main matmul family; same train-step program as v272 |
| `custom-call` | 29,396.8 ms | 28.1% | Splash attention; DKV no-residuals 14.8%, fwd residuals 13.1% |
| `loop fusion` | 10,715.8 ms | 10.2% | Memory-bound elementwise/update work |

## HLO Dump

- **Local copy**: `/tmp/qwen3-v273-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **SHA256**: `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
- **Size**: **19,915,660 bytes / 119,349 lines**
- **v272 comparison**: byte-identical (`cmp_v272=0`)
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

**Supported.** v273 exactly confirms the v272 RS3+AR2 selective-resources HLO
and profile family. Carry RS3+AR2 selective resources over the prior RS4+AR2
frontier: throughput remains in the same **43.3% MFU** band, XProf step time is
slightly better than v269 (**4358.1 ms** vs **4359.6 ms**), and the peak HBM
margin is much better (**0.3181 GiB** free vs v269 **0.0896 GiB** free).

The speed delta is too small to claim a meaningful MFU increase, but the memory
margin is now confirmed and should become the preferred base for the next
scheduler/latency probes.
