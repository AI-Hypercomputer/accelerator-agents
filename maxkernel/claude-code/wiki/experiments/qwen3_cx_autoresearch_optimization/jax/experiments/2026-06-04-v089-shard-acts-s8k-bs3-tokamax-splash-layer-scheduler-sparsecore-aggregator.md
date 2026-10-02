---
title: "Qwen3 JAX v089 - shard acts s8k bs3 tokamax splash layer scheduler sparse-core aggregator"
type: experiment
hypothesis: "Pairing the latency-hiding layer scheduler with sparse-core collective aggregation may make the layer-scheduler path valid and improve overlap."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported/provisional
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, latency-hiding, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, supported, provisional, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v089 - shard acts s8k bs3 tokamax splash layer scheduler sparse-core aggregator

## Hypothesis under test

**Hypothesis**: v088 showed that forcing
`--xla_tpu_enable_latency_hiding_layer_scheduler=true` alone is invalid because
the sparse-core collective aggregator is required. Enabling the required
aggregator explicitly may make the layer-scheduler path valid and improve
layer-level collective/compute overlap.

**Mechanism**: Keep the v074/v075 frontier fixed and add only the required pair:
`--xla_tpu_enable_latency_hiding_layer_scheduler=true` and
`--xla_tpu_enable_sparse_core_collective_aggregator=true`. Do not carry v076-v087
refuted/tie scheduler variants.

**Predicted signal**: Loss stays valid and throughput exceeds the confirmed
51,021 tok/s / 36.6% MFU band, or total profile time drops with no increase in
custom-call or data-formatting time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v089-layeragg`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v089-shard-acts-s8k-bs3-tokamax-splash-layer-scheduler-sparsecore-aggregator`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_latency_hiding_layer_scheduler=true` and
  `--xla_tpu_enable_sparse_core_collective_aggregator=true` to the v074/v075
  bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v089-layeragg` completed cleanly on 2026-06-04
with normal loss.

- Worker 0: 51,026 tok/s, 36.6% MFU
- Worker 1: 51,000 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is a new best-observed single-worker value by 1 tok/s over v078, but the
second worker is lower and the profile is effectively unchanged. Rerun before
treating it as durable.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v089-shard-acts-s8k-bs3-tokamax-splash-layer-scheduler-sparsecore-aggregator/2026_06_04_05_56_44`

Profile:

- MXU utilization: 56.8%
- Total time: 92,770.4 ms; idle 302.3 ms / 0.3%
- `jit_train_step`: 92,467.4 ms / 99.7%
- Convolution fusion: 45,706.6 ms / 49.3%
- Custom-call: 22,056.6 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,641.2 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,415.4 ms / 11.2%
- Loop fusion: 16,705.1 ms / 18.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v089-shard-acts-s8k-bs3-tokamax-splash-layer-scheduler-sparsecore-aggregator/hlo/`.

- HLO object count: 1,619
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,375,271 bytes
- Flagfile confirms:
  - `--xla_tpu_enable_latency_hiding_layer_scheduler='true'`
  - `--xla_tpu_enable_sparse_core_collective_aggregator='true'`
  - `--xla_tpu_enable_brkga_latency_hiding_scheduler=auto`
  - `--xla_tpu_enable_ilp_latency_hiding_scheduler=auto`
  - `--xla_tpu_enable_collective_compute_fusion=auto`
  - `--xla_tpu_async_collective_fusion_with_start_done_only=false`
  - `--xla_tpu_async_collective_fusion_fuse_multiple_collectives=false`
  - `--xla_tpu_enable_async_collective_fusion_fuse_kloop_fusions=false`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`

## Verdict

Supported/provisional. Pairing the layer scheduler with sparse-core collective
aggregation makes the v088 axis valid and produces a tiny new best-observed
single-worker throughput of 51,026 tok/s. The profile is still in the same
band, so this should be rerun before banking it as a durable frontier.
