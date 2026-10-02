---
title: "Qwen3 JAX v088 - shard acts s8k bs3 tokamax splash latency-hiding layer scheduler"
type: experiment
hypothesis: "Forcing the latency-hiding layer scheduler may improve layer-level overlap at the selective-resources frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, latency-hiding, layer-scheduler, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, invalid, startup-failure, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v088 - shard acts s8k bs3 tokamax splash latency-hiding layer scheduler

## Hypothesis under test

**Hypothesis**: The v074/v075 frontier leaves
`--xla_tpu_enable_latency_hiding_layer_scheduler=auto`. Forcing the layer
scheduler may improve overlap at the layer boundary while leaving the
previously tested multi-compute and dependent-collective subfeatures at auto.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_enable_latency_hiding_layer_scheduler=true`. Do not carry v076-v087
refuted/tie scheduler variants.

**Predicted signal**: Loss stays valid and throughput exceeds the confirmed
51,021 tok/s / 36.6% MFU band, or total profile time drops with no increase in
custom-call or data-formatting time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v088-layerlhs`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v088-shard-acts-s8k-bs3-tokamax-splash-latency-hiding-layer-scheduler`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_latency_hiding_layer_scheduler=true` to the v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v088-layerlhs` failed during startup before model
construction completed.

- Retained pod: `alekseyv-qwen3-v088-layerlhs-slice-job-0-0-5wwkh`
- Exit code: 1
- No train steps completed.

Failure signature:

```text
jax.errors.JaxRuntimeError: INVALID_ARGUMENT: Latency hiding layer scheduler requires sparse core collective aggregator to be enabled.
```

## Profile

No XProf run was produced because the workload failed before executing training
steps.

## HLO Dump

Only startup HLO artifacts were produced under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v088-shard-acts-s8k-bs3-tokamax-splash-latency-hiding-layer-scheduler/hlo/`.

- HLO object count: 17
- No train-step HLO was produced.

## Verdict

Invalid as a standalone flag. Forcing
`--xla_tpu_enable_latency_hiding_layer_scheduler=true` requires the sparse-core
collective aggregator to be enabled explicitly. A paired probe can test whether
`--xla_tpu_enable_sparse_core_collective_aggregator=true` makes this axis valid.
