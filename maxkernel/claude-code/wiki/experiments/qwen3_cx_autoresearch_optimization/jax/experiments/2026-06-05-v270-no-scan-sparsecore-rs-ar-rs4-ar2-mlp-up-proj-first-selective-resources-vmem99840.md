---
title: "Qwen3 JAX v270 - Selective-resources frontier with scoped VMEM 99840"
type: experiment
hypothesis: "The confirmed v269 selective-resources schedule is memory-tight at 31.16/31.25 GiB; a small scoped-VMEM downshift may preserve the 60.3k tok/s schedule while increasing allocator headroom."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs4-ar2-selective-resources-vmem99840
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, vmem, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v270 - Selective-resources frontier with scoped VMEM 99840

## Hypothesis under test

**Hypothesis**: v269 confirms the selective-resources frontier at **60,338
tok/s / 43.3% MFU** with **4359.6 ms / 68.2% MXU**, but XProf reports **31.16
GiB / 31.25 GiB** peak HBM and only **0.0896 GiB** free. Lowering
`--xla_tpu_scoped_vmem_limit_kib` from `100352` to `99840` may change scoped
VMEM allocation enough to recover headroom while preserving the new schedule
family.

**Mechanism**: Reuse v269 exactly, changing only
`--xla_tpu_scoped_vmem_limit_kib=99840`. Keep selective resources forced on,
RS latency multiplier `4`, AR latency multiplier `2`, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, Tokamax
Splash, MaxText CE, no-scan, remat, and activation sharding.

**Predicted signal**: Support requires clean finite loss and either the same
throughput band as v269 (**60.3k tok/s / 43.3% MFU**) with lower peak HBM, or a
small profile improvement without increasing memory. A byte-identical HLO with
lower memory would be ideal; a changed HLO must preserve the profile step-time
band near **4360 ms**.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, lower throughput/profile regression toward the v264
**59,954 tok/s / 4387.7 ms** band, unchanged memory with no speedup, or a
schedule that worsens the already-tight memory margin.

## Setup

GKE workload: `alekseyv-q3-v270-sr-vmem99840`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v270-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources-vmem99840`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduler/VMEM isolate only. Model code, params, math, random
fake-data setup, MLP up-proj-first image, and all runtime flags match v269
except `--xla_tpu_scoped_vmem_limit_kib`.

## Results

Workload `alekseyv-q3-v270-sr-vmem99840` completed cleanly on both workers
with normal fake-data loss and `EXIT_CODE=0`.

- Worker0: **60,116 tok/s / 7,514 tok/s/chip / 43.1% MFU** over steps 2-19.
- Worker1: **60,091 tok/s / 7,511 tok/s/chip / 43.1% MFU** over steps 2-19.
- Best observed step: worker0 step 15 at **60,229 tok/s**; worker1 step 15
  reached **60,225 tok/s**.
- Loss ended at **12.0463** on both workers.

This is a clean regression from the confirmed v269 frontier (**60,338 tok/s /
43.3% MFU**, best step **60,437 tok/s**).

## Profile

XProf profile:
`2026-06-05-qwen3-jax-v270-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources-vmem99840/2026_06_05_16_13_28`.

- Step time: **4374.8 ms**.
- MXU utilization: **67.9%**.
- Peak HBM: **31.13 GiB / 31.25 GiB**.
- Free HBM margin: **0.117 GiB**, up only **0.0274 GiB** versus v269.
- Op profile total: **105,023.4 ms** across hosts, with **284.1 ms idle**
  (0.3%).
- `jit_train_step`: **99.7%** of profile time.
- Largest op buckets: convolution fusion **57,295.4 ms** (54.6%),
  custom-call **29,402.5 ms** (28.0%), loop fusion **10,727.9 ms** (10.2%).

## HLO Dump

Optimized train-step HLO:
`/tmp/qwen3-v270-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- SHA256:
  `ed58458efdb96ad67a71a3ea3a837fc38446f66da69d5a50c9df0ef675c87caa`
- Size: **19,903,924 bytes / 119,288 lines**.
- Compared with v269: `cmp_v269=1`, changed HLO.
- Counts: `all-gather` 6988, `all-reduce` 165, `reduce-scatter` 2774,
  `async-start`/`async-done` 394/394, `convolution` 3083, `custom-call` 3849,
  `fusion` 30860, `copy` 22123, `tokamax` 2, `jvp_jit_silu__` 576.
- Flagfile confirmed the intended isolate:
  `--xla_tpu_scoped_vmem_limit_kib=99840` with
  `--xla_latency_hiding_scheduler_enable_selective_resources='true'`.

## Verdict

**Refuted.** Lowering scoped VMEM from `100352` to `99840` KiB does recover a
small amount of memory headroom, but the gain is too small to justify the
performance loss: free HBM improves only from **0.0896 GiB** to **0.117 GiB**,
while throughput falls to **60,116 tok/s** and XProf step time worsens to
**4374.8 ms**. Keep v269's `100352` KiB scoped VMEM setting on the frontier.
