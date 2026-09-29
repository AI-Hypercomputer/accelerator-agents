---
title: "Qwen3 JAX v096 - shard acts s8k bs3 tokamax splash sparse-core aggregator only"
type: experiment
hypothesis: "Sparse-core collective aggregation alone may explain the v089/v090 layeragg gain."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted/tie
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, refuted, tie, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v096 - shard acts s8k bs3 tokamax splash sparse-core aggregator only

## Hypothesis under test

**Hypothesis**: v089/v090 improved the prior v074/v075 scheduler frontier by
enabling the layer scheduler plus sparse-core collective aggregator pair. Since
v088 proved the layer scheduler alone is invalid, this run tests the other side
of the pair: sparse-core collective aggregation alone may be the actual source
of the small throughput gain.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_enable_sparse_core_collective_aggregator=true`. Do not enable
`--xla_tpu_enable_latency_hiding_layer_scheduler`.

**Predicted signal**: Loss stays valid and throughput exceeds the old
51,021 tok/s v074/v075 band, ideally matching the v089/v090 51,026+ repeated
band.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v096-aggonly`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v096-shard-acts-s8k-bs3-tokamax-splash-sparsecore-aggregator-only`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v074/v075 bundle plus
  `--xla_tpu_enable_sparse_core_collective_aggregator=true`; layer scheduler
  left auto/off.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v096-aggonly` completed cleanly on 2026-06-04 with
normal loss, but did not reproduce the v089/v090 layeragg gain.

- Worker 0: 51,011 tok/s, 36.6% MFU
- Worker 1: 50,989 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v096-shard-acts-s8k-bs3-tokamax-splash-sparsecore-aggregator-only/2026_06_04_07_11_26`

Profile:

- MXU utilization: 56.8%
- Total time: 92,799.7 ms; idle 268.5 ms / 0.3%
- `jit_train_step`: 92,530.6 ms / 99.7%
- Convolution fusion: 45,719.2 ms / 49.3%
- Custom-call: 22,057.0 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,641.1 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,415.8 ms / 11.2%
- Loop fusion: 16,715.4 ms / 18.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v096-shard-acts-s8k-bs3-tokamax-splash-sparsecore-aggregator-only/hlo/`.

- HLO object count: 1,619
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,375,271 bytes
- Flagfile confirms:
  - `--xla_tpu_enable_sparse_core_collective_aggregator='true'`
  - `--xla_tpu_enable_latency_hiding_layer_scheduler=auto`
  - `--xla_tpu_enable_brkga_latency_hiding_scheduler=auto`
  - `--xla_tpu_enable_ilp_latency_hiding_scheduler=auto`
  - `--xla_tpu_enable_layer_scheduler_for_dependent_collectives=auto`
  - `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=auto`
  - `--xla_latency_hiding_scheduler_resource_serializing=false`
  - `--xla_tpu_enable_collective_compute_fusion=auto`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`

## Verdict

Refuted/tie. Sparse-core collective aggregation alone is valid, but it does not
match the v089/v090 repeated 51,026+ tok/s band and falls below even the old
v074/v075 51,021 tok/s supported band. The small v089/v090 gain requires the
explicit layer-scheduler plus sparse-core-aggregator pair.
