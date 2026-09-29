---
title: "Qwen3 JAX v272 - Selective-resources frontier with RS3+AR2"
type: experiment
hypothesis: "After v271 closed RS5 under selective resources due to HBM failure, lowering only the SparseCore reduce-scatter latency multiplier from 4 to 3 may preserve the selective-resources speedup with less memory pressure or a better low-side schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs3-ar2-selective-resources
status: supported-provisional
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, latency, rs3, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v272 - Selective-resources frontier with RS3+AR2

## Hypothesis under test

**Hypothesis**: v269 confirms RS4+AR2 with selective resources as the current
frontier (**60,338 tok/s / 43.3% MFU**, **4359.6 ms / 68.2% MXU**), while
v271 shows RS5 under selective resources cannot load due to HBM pressure.
Lowering only `--xla_tpu_sparse_core_reduce_scatter_latency_multiplier` from
`4` to `3` may preserve selective-resource scheduling benefits with less HBM
pressure or select a better low-side schedule.

**Mechanism**: Reuse v269 exactly, changing only RS latency multiplier from
`4` to `3`. Keep selective resources forced on, AR latency multiplier `2`,
scoped VMEM `100352`, SparseCore reduce-scatter/all-reduce offload,
SparseCore collective aggregator, Tokamax Splash, MaxText CE, no-scan, remat,
and activation sharding.

**Predicted signal**: Support requires clean finite loss and either throughput
above v269's confirmed **60,338 tok/s / 43.3% MFU**, or same-band throughput
with XProf step time at or below **4359.6 ms** and improved HBM headroom.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, throughput/profile regression toward the old RS3 non-SR band,
or memory unchanged/worse with no speedup.

## Setup

GKE workload: `alekseyv-q3-v272-sr-rs3-ar2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v272-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduler isolate only. Model code, params, math, random fake-data
setup, MLP up-proj-first image, and all runtime flags match v269 except
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier`.

## Results

Workload `alekseyv-q3-v272-sr-rs3-ar2` completed cleanly on both slices with
normal loss and `EXIT_CODE=0`.

| Worker | Avg throughput, steps 2-19 | Tok/s/chip | Approx MFU | Best observed post-profile step |
|--------|----------------------------|------------|------------|---------------------------------|
| 0 | **60,318 tok/s** | **7,540** | **43.2%** | 60,451 tok/s |
| 1 | **60,351 tok/s** | **7,544** | **43.3%** | 60,454 tok/s |

Compared with confirmed v269 RS4+AR2 selective resources, v272 is throughput
same-band: worker0 matches v269's worker0 average, worker1 is +13 tok/s over
v269's best worker average, and MFU rounds to the same **43.3%**. The stronger
signal is memory: v272 lowers peak HBM materially while preserving the same
wall-time band.

## Profile

- **Source**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v272-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources`
- **xprof run**: `2026-06-05-qwen3-jax-v272-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources/2026_06_05_16_39_14`
- **Average step time**: **4358.2 ms**
- **MXU utilization**: **68.1%**
- **Peak HBM**: **30.93 GiB / 31.25 GiB**
- **Free HBM at peak**: **0.3181 GiB**
- **Stack reservation / heap allocation**: **23.0798 GiB / 7.8482 GiB**
- **Idle**: **0.3%** of op-profile total time

Top op-profile buckets for `jit_train_step`:

| Bucket | Time | Share | Notes |
|--------|------|-------|-------|
| `convolution fusion` | 56,835.8 ms | 54.3% | Main matmul family; top duplicated fusions include AG+conv+Silu neighborhoods |
| `custom-call` | 29,397.1 ms | 28.1% | Splash attention; DKV no-residuals 14.8%, fwd residuals 13.1% |
| `loop fusion` | 10,722.7 ms | 10.2% | Memory-bound elementwise/update work, HBM BW util up to ~0.92 in top fusions |

## HLO Dump

- **Local copy**: `/tmp/qwen3-v272-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **SHA256**: `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
- **Size**: **19,915,660 bytes / 119,349 lines**
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

**Supported, provisional memory-headroom tie.** RS3+AR2 under selective
resources preserves the confirmed v269 throughput band and slightly improves
the XProf step-time signal (**4358.2 ms** vs v269 **4359.6 ms**) while raising
free HBM from **0.0896 GiB** to **0.3181 GiB**. The throughput gain is too small
to call a new performance frontier from one run, but the memory margin is large
enough to justify an exact rerun before choosing whether RS3+AR2 should replace
RS4+AR2 as the carried selective-resources setting.

Next: exact rerun of v272. If the rerun confirms same-band throughput with the
same HBM relief, carry RS3+AR2 selective resources as the safer frontier and
reopen previously memory-blocked probes on top of it.
