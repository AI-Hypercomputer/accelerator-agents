---
title: "Qwen3 JAX v090 - shard acts s8k bs3 tokamax splash layer scheduler sparse-core aggregator rerun"
type: experiment
hypothesis: "Rerunning v089 will show whether the layer-scheduler plus sparse-core aggregator pair is a durable improvement or just noise."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, latency-hiding, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, rerun, supported, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v090 - shard acts s8k bs3 tokamax splash layer scheduler sparse-core aggregator rerun

## Hypothesis under test

**Hypothesis**: v089 produced a tiny new best-observed worker value of
51,026 tok/s with the layer scheduler plus sparse-core collective aggregator
pair. An exact rerun should determine whether that result is durable or just
run-to-run noise.

**Mechanism**: Exactly rerun v089: keep the v074/v075 frontier fixed and add
`--xla_tpu_enable_latency_hiding_layer_scheduler=true` plus
`--xla_tpu_enable_sparse_core_collective_aggregator=true`. Do not carry v076-v087
refuted/tie scheduler variants.

**Predicted signal**: Loss stays valid and throughput repeats above the
confirmed 51,021 tok/s / 36.6% MFU band, ideally near or above v089's
51,026 tok/s worker value.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion back in the v074/v075 band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v090-layeragg-rerun`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v090-shard-acts-s8k-bs3-tokamax-splash-layer-scheduler-sparsecore-aggregator-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v074/v075 bundle plus
  `--xla_tpu_enable_latency_hiding_layer_scheduler=true` and
  `--xla_tpu_enable_sparse_core_collective_aggregator=true`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v090-layeragg-rerun` completed cleanly on 2026-06-04
with normal loss.

- Worker 0: 51,033 tok/s, 36.6% MFU
- Worker 1: 51,005 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This confirms the v089 paired layer-scheduler plus sparse-core-aggregator
variant as a supported scheduler frontier. The exact best-worker value is tiny
and noisy, but v089/v090 both repeat above the old 51,021 tok/s confirmed band.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v090-shard-acts-s8k-bs3-tokamax-splash-layer-scheduler-sparsecore-aggregator-rerun/2026_06_04_06_03_39`

Profile:

- MXU utilization: 56.8%
- Total time: 92,767.7 ms; idle 293.8 ms / 0.3%
- `jit_train_step`: 92,473.2 ms / 99.7%
- Convolution fusion: 45,704.8 ms / 49.3%
- Custom-call: 22,056.4 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,641.2 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,415.2 ms / 11.2%
- Loop fusion: 16,715.6 ms / 18.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

No HLO objects were written under the v090 run path, consistent with a
persistent-cache hit from v089.

Use the v089 HLO dump for this exact compiled program:

- HLO object count: 1,619
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,375,271 bytes
- v089 flagfile confirms:
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

Supported. The v089/v090 pair confirms that adding
`--xla_tpu_enable_latency_hiding_layer_scheduler=true` together with
`--xla_tpu_enable_sparse_core_collective_aggregator=true` is valid and repeats
slightly above the prior 51,021 tok/s band. Adopt this paired flag stack as the
new scheduler frontier, while treating the exact 51,033 tok/s high-water mark as
small/noisy.
