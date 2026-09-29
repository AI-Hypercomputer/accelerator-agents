---
title: "Qwen3 JAX v268 - MLP up projection first with RS4+AR2 and selective resources"
type: experiment
hypothesis: "The confirmed RS4+AR2 HLO currently has latency-hiding selective resources disabled; forcing selective resources may improve overlap/resource assignment on the new MLP up-proj-first schedule without changing math or collective placement."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs4-ar2-selective-resources
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v268 - MLP up projection first with RS4+AR2 and selective resources

## Hypothesis under test

**Hypothesis**: The v264 flagfile shows
`--xla_latency_hiding_scheduler_enable_selective_resources='false'` on the
confirmed MLP up-proj-first + RS4+AR2 frontier. Earlier HLO families saw
selective-resource toggles matter, but this new HLO family has not been tested
with selective resources forced on. Enabling it may improve overlap/resource
assignment without changing model semantics.

**Mechanism**: Reuse the v264 runtime stack and image, keep RS latency
multiplier `4`, AR latency multiplier `2`, layer scheduler disabled/auto, and
add only `--xla_latency_hiding_scheduler_enable_selective_resources=true`.
Keep Tokamax Splash, MaxText CE, no-scan, remat, activation sharding,
SparseCore reduce-scatter and all-reduce offload, the SparseCore collective
aggregator, and all-gather on async collective fusion.

**Predicted signal**: Support requires clean finite loss and either throughput
above v264's best full-window **59,954 tok/s / 43.0% MFU**, or the same
throughput band with XProf step time below v264's **4387.7 ms** and unchanged
or lower HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, HBM pressure above the current **30.94 GiB** band with no
speedup, or a clean same/lower throughput/profile result.

## Setup

GKE workload: `alekseyv-q3-v268-mlp-up-rs4-ar2-sr`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v268-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduler-flag isolate only. Model code, params, math, random fake-data
setup, and MLP up-proj-first image match v263-v267.

## Results

Workload `alekseyv-q3-v268-mlp-up-rs4-ar2-sr` completed cleanly on both
workers with normal fake-data loss and `EXIT_CODE=0`.

- Worker0: **60,315 tok/s / 7,539 tok/s/chip / 43.2% MFU** over steps 2-19.
- Worker1: **60,336 tok/s / 7,542 tok/s/chip / 43.3% MFU** over steps 2-19.
- Best observed step: worker1 step 15 at **60,450 tok/s**; worker0 step 15
  reached **60,449 tok/s**.
- Loss ended in the expected band around **12.0458**.

This is a meaningful primary-metric lift over v264's confirmed best-worker
full-window **59,954 tok/s / 43.0% MFU** (+382 tok/s, about +0.64%) and also
improves the best observed single-step high-water mark from **60,066 tok/s** to
**60,450 tok/s**.

## Profile

XProf profile:
`2026-06-05-qwen3-jax-v268-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources/2026_06_05_15_43_13`.

- Step time: **4359.7 ms**.
- MXU utilization: **68.2%**.
- Peak HBM: **31.16 GiB / 31.25 GiB**.
- Free HBM margin: about **0.09 GiB**, so this point is valid but memory-tight.
- Op profile total: **104,645 ms** across hosts, with **273.8 ms idle** (0.3%).
- `jit_train_step`: **99.7%** of profile time.
- Largest op buckets: convolution fusion **56,880.6 ms** (54.4%),
  custom-call **29,399.1 ms** (28.1%), loop fusion **10,723.7 ms** (10.2%).

## HLO Dump

Optimized train-step HLO:
`/tmp/qwen3-v268-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- SHA256:
  `a92469ed51218d7814d863f5e78b678967b81a836d9a558a1d9c966cc4a206e0`
- Size: **19,901,683 bytes / 119,264 lines**.
- Compared with v264: `cmp_v264=1`, so this is a real new schedule.
- Counts: `all-gather` 6988, `all-reduce` 165, `reduce-scatter` 2774,
  `async-start`/`async-done` 391/391, `convolution` 3083, `custom-call` 3846,
  `fusion` 30859, `copy` 22124, `tokamax` 2, `jvp_jit_silu__` 576.
- Flagfile confirmed the intended isolate:
  `--xla_latency_hiding_scheduler_enable_selective_resources='true'` with
  SparseCore collective aggregator still enabled, RS latency multiplier `4`,
  and AR latency multiplier `2`.

## Verdict

**Provisional supported; confirmed by v269.** Forcing selective resources on
the confirmed MLP up-proj-first + RS4+AR2 frontier selected a new HLO and
improved both train throughput and XProf step time. v269 exactly reproduced
this HLO/profile band, so selective resources is now carried as a confirmed
frontier flag. The caveat is that the profile is very close to HBM capacity
(**31.16 GiB / 31.25 GiB**).
