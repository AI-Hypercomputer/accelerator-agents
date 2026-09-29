---
title: "Qwen3 JAX v246 - RS2+AR2 no TPU scheduler memory-pressure tracking"
type: experiment
hypothesis: "The post-SparseCore RS2+AR2 frontier may no longer need TPU scheduler memory-pressure tracking, and disabling it could let the scheduler preserve more overlap without reintroducing HBM pressure."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs2-ar2-no-mem-pressure-tracking
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, scheduler, memory-pressure, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v246 - RS2+AR2 no TPU scheduler memory-pressure tracking

## Hypothesis under test

**Hypothesis**: Earlier no-memory-pressure-tracking probes were refuted before
SparseCore RS/AR offload became the frontier. With RS/AR offloaded and VMEM
retuning closed, disabling TPU scheduler memory-pressure tracking may reduce
over-conservative scheduling while SparseCore collective placement preserves
HBM fit.

**Mechanism**: Keep the material v230 RS2+AR2 SparseCore frontier fixed except
change `--xla_tpu_enable_scheduler_memory_pressure_tracking=true` to `false`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band (**59,749-59,750 tok/s**) or a same-band run with lower
XProf step time/peak HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss,
byte-identical HLO with no throughput/profile improvement, changed HLO with a
worse profile, or clean completion at or below the current frontier band.

## Setup

GKE workload `alekseyv-q3-v246-nomemtrack`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v246-no-scan-sparsecore-rs-ar-rs2-ar2-no-mem-pressure-tracking`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA scheduler flag isolate and preserves
model math.

## Result

Workload `alekseyv-q3-v246-nomemtrack` completed cleanly with `EXIT_CODE=0`
on both workers and normal loss ending at 12.0459.

- Worker0: **59,732 tok/s** average over steps 2-19, **59,878 tok/s** best
  step, **59,767 tok/s** average over steps 15-19, **59,744 tok/s** excluding
  profiled steps.
- Worker1: **59,712 tok/s** average over steps 2-19, **59,878 tok/s** best
  step, **59,767 tok/s** average over steps 15-19, **59,744 tok/s** excluding
  profiled steps.

The best-worker primary full-window average is below the v230/v236 frontier
band (**59,749-59,750 tok/s**), although the late/no-trace windows remain close.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v246-no-scan-sparsecore-rs-ar-rs2-ar2-no-mem-pressure-tracking/2026_06_05_10_24_02`

- Step time: **4401.5 ms**
- MXU utilization: **67.5%**
- Peak HBM: **30.93 GiB** (**23.0798 GiB** stack, **7.8487 GiB** heap,
  **0.3175 GiB** free)
- Idle: **0.3%**

Top HLO op buckets:

- convolution fusion: **56,771.9 ms** / **53.7%**
- custom-call: **29,411.5 ms** / **27.8%**
- loop fusion: **10,701.2 ms** / **10.1%**
- custom fusion: **3,308.5 ms** / **3.1%**
- data formatting: **2,909.5 ms** / **2.8%**

The profile matches the v230/v242 same-HLO band rather than improving it.

## HLO Dump

Optimized train-step HLO:

- Local dump:
  `/tmp/qwen3-v246-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`
- Size: **19,925,715 bytes**
- Lines: **119,437**

The HLO is byte-identical to v230/v242.

## Verdict

**Refuted / same-HLO tie.** Disabling TPU scheduler memory-pressure tracking
on the post-SparseCore RS2+AR2 frontier preserves the v230 optimized train-step
HLO and profile band, but it does not improve throughput, step time, or HBM.
Keep `--xla_tpu_enable_scheduler_memory_pressure_tracking=true` on the carried
frontier.
