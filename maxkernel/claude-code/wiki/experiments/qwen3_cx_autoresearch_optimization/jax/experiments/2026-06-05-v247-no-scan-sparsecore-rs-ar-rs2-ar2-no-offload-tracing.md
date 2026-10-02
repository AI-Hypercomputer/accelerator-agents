---
title: "Qwen3 JAX v247 - RS2+AR2 no SparseCore offload tracing"
type: experiment
hypothesis: "The material RS2+AR2 frontier may not need active SparseCore RS/AR offload tracing, and disabling it could reduce profile/runtime noise without changing the optimized train-step HLO."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs2-ar2-no-offload-tracing
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, tracing, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v247 - RS2+AR2 no SparseCore offload tracing

## Hypothesis under test

**Hypothesis**: v238 showed disabling active SparseCore offload tracing is a
same-HLO diagnostics-size knob on the RS3+AR2 same-band branch. The material
RS2+AR2 v230 frontier still has RS/AR tracing enabled; disabling it may reduce
measurement/runtime noise on the carried HLO without changing collective
placement.

**Mechanism**: Keep v230 fixed except change
`--xla_tpu_enable_reduce_scatter_offload_tracing=true` and
`--xla_tpu_enable_all_reduce_offload_tracing=true` to `false`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band (**59,749-59,750 tok/s**) or a same-band run with lower
profile artifact size/step time.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with worse profile, or clean completion at or below the v230/v236 band.

## Setup

GKE workload `alekseyv-q3-v247-rs2ar2-notrace`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v247-no-scan-sparsecore-rs-ar-rs2-ar2-no-offload-tracing`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is a tracing flag isolate and preserves model
math.

## Result

Workload `alekseyv-q3-v247-rs2ar2-notrace` completed cleanly with
`EXIT_CODE=0` on both workers and normal loss ending at 12.0459.

- Worker0: **59,748 tok/s** average over steps 2-19, **59,895 tok/s** best
  step, **59,795 tok/s** average over steps 15-19, **59,763 tok/s** excluding
  profiled steps.
- Worker1: **59,721 tok/s** average over steps 2-19, **59,884 tok/s** best
  step, **59,797 tok/s** average over steps 15-19, **59,759 tok/s** excluding
  profiled steps.

This ties the frontier band in late/no-trace windows but does not beat the
v230/v236 primary full-window band (**59,749-59,750 tok/s**).

## Profile

XProf run:
`2026-06-05-qwen3-jax-v247-no-scan-sparsecore-rs-ar-rs2-ar2-no-offload-tracing/2026_06_05_10_36_52`

- Step time: **4402.2 ms**
- MXU utilization: **67.5%**
- Peak HBM: **30.93 GiB** (**23.0798 GiB** stack, **7.8487 GiB** heap,
  **0.3175 GiB** free)

Top HLO op buckets:

- convolution fusion: **56,766.4 ms** / **53.7%**
- custom-call: **29,411.4 ms** / **27.8%**
- loop fusion: **10,703.8 ms** / **10.1%**
- custom fusion: **3,311.5 ms** / **3.1%**
- data formatting: **2,909.7 ms** / **2.8%**

## HLO Dump

Optimized train-step HLO:

- Local dump:
  `/tmp/qwen3-v247-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`
- Size: **19,925,715 bytes**
- Lines: **119,437**

The HLO is byte-identical to v230/v242/v246.

## Verdict

**Refuted / same-HLO tie.** Disabling active SparseCore RS/AR offload tracing
on the material RS2+AR2 HLO does not improve the primary full-window average,
profile step time, or memory footprint. Treat offload tracing as a
diagnostics-size knob; keep the v230/v236 frontier unchanged.
