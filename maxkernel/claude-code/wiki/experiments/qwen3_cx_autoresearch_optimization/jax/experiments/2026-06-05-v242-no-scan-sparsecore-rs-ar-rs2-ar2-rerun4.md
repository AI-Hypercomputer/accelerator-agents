---
title: "Qwen3 JAX v242 - RS2+AR2 scheduler rerun4"
type: experiment
hypothesis: "After VMEM retuning failed on both post-SparseCore HLO families, a different latency-hiding scheduler rerun count may find a better ordering for the material RS2+AR2 frontier without changing math or collective placement."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs2-ar2-rerun4
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, scheduler, latency-hiding, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v242 - RS2+AR2 scheduler rerun4

## Hypothesis under test

**Hypothesis**: The material v230 RS2+AR2 frontier uses
`--xla_latency_hiding_scheduler_rerun=3`. A fourth scheduler rerun may find a
slightly better ordering for the same SparseCore offload split now that VMEM
retuning is closed.

**Mechanism**: Keep v230 fixed except change
`--xla_latency_hiding_scheduler_rerun=3` to `4`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band (**59,749-59,750 tok/s**) or a same-band run with lower
profile step time/peak HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss,
byte-identical HLO with no throughput/profile improvement, changed HLO with
worse profile, or clean completion at or below the v230/v236 band.

## Setup

GKE workload `alekseyv-q3-v242-rerun4`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v242-no-scan-sparsecore-rs-ar-rs2-ar2-rerun4`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA scheduler-search flag isolate and
preserves model math.

## Result

Workload `alekseyv-q3-v242-rerun4` completed cleanly with `EXIT_CODE=0` on
both workers and normal loss ending at 12.0459.

- Worker0: **59,740 tok/s** average over steps 2-19, **59,906 tok/s** best
  step, **59,781 tok/s** average over steps 15-19, **59,750 tok/s** excluding
  profiled steps.
- Worker1: **59,703 tok/s** average over steps 2-19, **59,907 tok/s** best
  step, **59,780 tok/s** average over steps 15-19, **59,752 tok/s** excluding
  profiled steps.

The best-worker primary full-window average is below the v230/v236 band
(**59,749-59,750 tok/s**), while the late and no-trace windows tie it within
noise.

## Profile

XProf split the two hosts into adjacent one-host runs:

- `2026-06-05-qwen3-jax-v242-no-scan-sparsecore-rs-ar-rs2-ar2-rerun4/2026_06_05_09_24_51`:
  **4400.4 ms** step time, **67.9% MXU**.
- `2026-06-05-qwen3-jax-v242-no-scan-sparsecore-rs-ar-rs2-ar2-rerun4/2026_06_05_09_24_50`:
  **4400.4 ms** step time, **66.9% MXU**.

Representative memory profile: **30.93 GiB** peak HBM, **23.0798 GiB** stack,
**7.8487 GiB** heap, **0.3175 GiB** free.

Representative one-host op profile:

- convolution fusion: **28,786.9 ms** / **54.6%**
- custom-call: **14,709.0 ms** / **27.9%**
- loop fusion: **5,383.0 ms** / **10.2%**

## HLO Dump

Optimized train-step HLO:

- Local dump:
  `/tmp/qwen3-v242-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`
- Size: **19,925,715 bytes**
- Lines: **119,437**

The HLO is byte-identical to v230.

## Verdict

**Refuted / same-HLO tie.** `--xla_latency_hiding_scheduler_rerun=4` preserves
the v230 optimized train-step HLO and profile band, but it does not materially
beat the confirmed v230/v236 frontier. Keep rerun `3` as the carried setting
unless a later experiment proves the rerun count interacts with another
structural change.
