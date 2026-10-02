---
title: "Qwen3 JAX v269 - Rerun MLP up projection first with RS4+AR2 and selective resources"
type: experiment
hypothesis: "The v268 selective-resources schedule is a real throughput improvement over the v264 confirmed frontier; an exact rerun should reproduce the same schedule family and the 60.3k tok/s / 43.3% MFU band."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs4-ar2-selective-resources-rerun
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, rerun, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v269 - Rerun MLP up projection first with RS4+AR2 and selective resources

## Hypothesis under test

**Hypothesis**: v268 forced selective resources on top of the confirmed v264
MLP up-proj-first + RS4+AR2 stack and produced a new provisional high-water
mark: **60,336 tok/s / 43.3% MFU** with **4359.7 ms / 68.2% MXU** in XProf.
Because v268 peak HBM was very tight (**31.16 GiB / 31.25 GiB**), an exact
rerun is required before carrying selective resources as a confirmed frontier
flag.

**Mechanism**: Exact rerun of v268. Same image, command, fake-data setup,
Tokamax Splash, MaxText CE, no-scan, remat, activation sharding, scoped VMEM
100352 KiB, SparseCore reduce-scatter/all-reduce offload, SparseCore
collective aggregator, RS latency multiplier `4`, AR latency multiplier `2`,
and `--xla_latency_hiding_scheduler_enable_selective_resources=true`.

**Predicted signal**: Support requires clean finite loss and reproduction of
the v268 schedule family and throughput band: ideally the same optimized HLO
SHA256
`a92469ed51218d7814d863f5e78b678967b81a836d9a558a1d9c966cc4a206e0`,
best-worker full-window throughput around **60.3k tok/s / 43.3% MFU**, and
profile step time near or below **4365 ms**.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, OOM or HBM allocator instability, a materially different slow
HLO, or clean throughput/profile regression back to the v264 **59,954 tok/s /
4387.7 ms** band.

## Setup

GKE workload: `alekseyv-q3-v269-mlp-up-rs4-ar2-sr-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v269-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Exact runtime rerun. Model code, params, math, random fake-data setup, MLP
up-proj-first image, and all runtime flags match v268.

## Results

Workload `alekseyv-q3-v269-mlp-up-rs4-ar2-sr-r` completed cleanly on both
workers with normal fake-data loss and `EXIT_CODE=0`.

- Worker0: **60,318 tok/s / 7,540 tok/s/chip / 43.2% MFU** over steps 2-19.
- Worker1: **60,338 tok/s / 7,542 tok/s/chip / 43.3% MFU** over steps 2-19.
- Best observed step: worker0 step 15 at **60,437 tok/s**; worker1 step 15
  reached **60,436 tok/s**.
- Loss ended at **12.0459** on both workers.

This reproduces v268's **60,315-60,336 tok/s / 43.2-43.3% MFU** band and
stays materially above the previous confirmed v264 frontier (**59,954 tok/s /
43.0% MFU**).

## Profile

XProf profile:
`2026-06-05-qwen3-jax-v269-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources-rerun/2026_06_05_15_58_24`.

- Step time: **4359.6 ms**.
- MXU utilization: **68.2%**.
- Peak HBM: **31.16 GiB / 31.25 GiB**.
- Free HBM margin: **0.0896 GiB**; utilization **99.71%**.
- Op profile total: **104,655.7 ms** across hosts, with **282.7 ms idle**
  (0.3%).
- `jit_train_step`: **99.7%** of profile time.
- Largest op buckets: convolution fusion **56,883.5 ms** (54.4%),
  custom-call **29,399.1 ms** (28.1%), loop fusion **10,722.4 ms** (10.2%).

## HLO Dump

Optimized train-step HLO:
`/tmp/qwen3-v269-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- SHA256:
  `a92469ed51218d7814d863f5e78b678967b81a836d9a558a1d9c966cc4a206e0`
- Size: **19,901,683 bytes / 119,264 lines**.
- Compared with v268: `cmp_v268=0`, byte-identical.
- Counts: `all-gather` 6988, `all-reduce` 165, `reduce-scatter` 2774,
  `async-start`/`async-done` 391/391, `convolution` 3083, `custom-call` 3846,
  `fusion` 30859, `copy` 22124, `tokamax` 2, `jvp_jit_silu__` 576.
- Flagfile confirmed the intended isolate:
  `--xla_latency_hiding_scheduler_enable_selective_resources='true'` with
  SparseCore collective aggregator still enabled, scoped VMEM `100352` KiB, RS
  latency multiplier `4`, and AR latency multiplier `2`.

## Verdict

**Confirmed supported.** v269 exactly reproduces the v268 selective-resources
HLO and profile band, and confirms selective resources as a carried frontier
flag on top of MLP up-proj-first + RS4+AR2. Current confirmed native-JAX best
is **60,338 tok/s / 7,542 tok/s/chip / 43.3% MFU** with **4359.6 ms / 68.2%
MXU** in XProf. The caveat is memory: peak HBM remains **31.16 GiB / 31.25
GiB**, so subsequent probes should avoid mechanisms that increase live memory
and may test very small scoped-VMEM downshifts.
