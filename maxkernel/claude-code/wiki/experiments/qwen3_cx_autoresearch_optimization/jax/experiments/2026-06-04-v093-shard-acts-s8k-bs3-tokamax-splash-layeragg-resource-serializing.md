---
title: "Qwen3 JAX v093 - shard acts s8k bs3 tokamax splash layeragg resource serializing"
type: experiment
hypothesis: "Resource serializing in the latency-hiding scheduler may improve the layer-scheduler plus sparse-core-aggregator frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted/tie
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, latency-hiding, layer-scheduler, sparse-core-collective-aggregator, resource-serializing, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, refuted, tie, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v093 - shard acts s8k bs3 tokamax splash layeragg resource serializing

## Hypothesis under test

**Hypothesis**: The v089/v090 layer-scheduler plus sparse-core-aggregator
frontier still leaves the same custom-call and loop-fusion mix as earlier
frontiers. Enabling latency-hiding scheduler resource serializing may reduce
resource contention in the schedule without changing model code.

**Mechanism**: Keep the v090 frontier fixed and add only
`--xla_latency_hiding_scheduler_resource_serializing=true`. Do not carry BRKGA
from v091 or ILP from v092.

**Predicted signal**: Loss stays valid and throughput exceeds the v090
51,033 tok/s best-observed worker value or at least repeats the 51,026+
supported band with a profile win.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v089/v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v093-layerresserial`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v093-shard-acts-s8k-bs3-tokamax-splash-layeragg-resource-serializing`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v090 bundle plus
  `--xla_latency_hiding_scheduler_resource_serializing=true`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v093-layerresserial` completed cleanly on 2026-06-04
with normal loss, but regressed below the v089/v090 frontier.

- Worker 0: 50,977 tok/s, 36.6% MFU
- Worker 1: 50,961 tok/s, 36.5% MFU
- Final loss: 12.0510 on worker 0, 12.0510 on worker 1
- Exit code: 0

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v093-shard-acts-s8k-bs3-tokamax-splash-layeragg-resource-serializing/2026_06_04_06_38_36`

Profile:

- MXU utilization: 56.8%
- Total time: 92,810.4 ms; idle 239.0 ms / 0.3%
- `jit_train_step`: 92,570.8 ms / 99.7%
- Convolution fusion: 45,612.7 ms / 49.1%
- Custom-call: 22,061.0 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,644.9 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,416.1 ms / 11.2%
- Loop fusion: 16,770.2 ms / 18.1%
- Peak HBM: 28.03 GiB of 31.25 GiB; 3.22 GiB free; 0.05% fragmentation

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v093-shard-acts-s8k-bs3-tokamax-splash-layeragg-resource-serializing/hlo/`.

- HLO object count: 1,619
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,303,468 bytes
- Flagfile confirms:
  - `--xla_tpu_enable_latency_hiding_layer_scheduler='true'`
  - `--xla_tpu_enable_sparse_core_collective_aggregator='true'`
  - `--xla_latency_hiding_scheduler_resource_serializing=true`
  - `--xla_tpu_enable_brkga_latency_hiding_scheduler=auto`
  - `--xla_tpu_enable_ilp_latency_hiding_scheduler=auto`
  - `--xla_tpu_enable_collective_compute_fusion=auto`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`

## Verdict

Refuted/tie. Resource serializing is valid and reduces peak HBM by about
0.29 GiB versus v090, but it worsens wall time and does not improve the
custom-call profile. Do not carry resource serializing forward for the
throughput frontier.
