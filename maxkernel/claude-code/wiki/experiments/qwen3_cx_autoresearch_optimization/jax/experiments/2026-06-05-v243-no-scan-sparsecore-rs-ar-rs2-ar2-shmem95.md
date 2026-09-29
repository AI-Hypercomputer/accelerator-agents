---
title: "Qwen3 JAX v243 - RS2+AR2 scheduler shared-memory limit 95"
type: experiment
hypothesis: "After the post-SparseCore RS2+AR2 frontier saturated VMEM and rerun-count probes, a modest scheduler shared-memory cap may force a better ordering around custom-call scratch without changing math or collective placement."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs2-ar2-shmem95
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, scheduler, shared-memory, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v243 - RS2+AR2 scheduler shared-memory limit 95

## Hypothesis under test

**Hypothesis**: The material v230 RS2+AR2 frontier leaves only about 0.32 GiB
free at peak HBM. A mild shared-memory scheduler cap may reduce custom-call
scratch pressure enough to improve the schedule, even though the same cap was
neutral before the SparseCore RS/AR latency-model changes.

**Mechanism**: Keep v230 fixed except change
`--xla_tpu_scheduler_percent_shared_memory_limit=100` to `95`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band (**59,749-59,750 tok/s**) or a same-band run with lower
profile step time/peak HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss,
byte-identical HLO with no throughput/profile improvement, changed HLO with
worse profile, or clean completion at or below the v230/v236 band.

## Setup

GKE workload `alekseyv-q3-v243-shmem95`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v243-no-scan-sparsecore-rs-ar-rs2-ar2-shmem95`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA scheduler shared-memory flag isolate
and preserves model math.

## Result

Workload `alekseyv-q3-v243-shmem95` completed cleanly with `EXIT_CODE=0` on
both workers and normal loss ending at 12.0459.

- Worker0: **59,317 tok/s** average over steps 2-19, **59,502 tok/s** best
  step, **59,322 tok/s** average over steps 15-19, **59,363 tok/s** excluding
  profiled steps.
- Worker1: **59,342 tok/s** average over steps 2-19, **59,500 tok/s** best
  step, **59,322 tok/s** average over steps 15-19, **59,361 tok/s** excluding
  profiled steps.

Best full-window throughput was **59,342 tok/s**, below v230/v236's
**59,749-59,750 tok/s** frontier band. Nominal MFU is about **42.6%**.

## Profile

XProf run
`2026-06-05-qwen3-jax-v243-no-scan-sparsecore-rs-ar-rs2-ar2-shmem95/2026_06_05_09_37_52`
shows:

- Step time: **4432.6 ms**
- MXU utilization: **66.9%**
- Peak HBM: **30.93 GiB** / 31.25 GiB
- Stack reservation: **23.0798 GiB**
- Heap allocation: **7.8487 GiB**
- Free memory at peak: **0.3175 GiB**

The aggregate op profile is still dominated by train-step convolution fusion
(**56,852.2 ms**, **53.4%** of profiled time). Idle time is low
(**313.6 ms**, **0.3%**), so the regression is schedule quality rather than a
host-side stall.

## HLO Dump

Optimized train-step HLO:

- Local dump:
  `/tmp/qwen3-v243-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`
- Size: **19,925,715 bytes**
- Lines: **119,437**

The HLO is byte-identical to v230 and v242.

## Verdict

**Refuted.** Lowering `--xla_tpu_scheduler_percent_shared_memory_limit` from
100 to 95 preserves the optimized HLO and memory footprint but worsens runtime
schedule quality: throughput drops to **59,342 tok/s** and profiled step time
worsens to **4432.6 ms** versus v230's **4400.7 ms**. Do not carry shmem95 on
the post-SparseCore RS2+AR2 frontier.
