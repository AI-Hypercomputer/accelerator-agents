---
title: "Qwen3 JAX v081 - shard acts s8k bs3 tokamax splash collective compute fusion"
type: experiment
hypothesis: "Forcing TPU collective-compute fusion may improve overlap around all-gather-adjacent compute fusions in the frontier schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, scheduler, collective-compute-fusion, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v081 - shard acts s8k bs3 tokamax splash collective compute fusion

## Hypothesis under test

**Hypothesis**: The frontier HLO/profile has many all-gather-adjacent
convolution fusions and a persistent 1% collective-permute-start bucket.
Forcing `--xla_tpu_enable_collective_compute_fusion=true` may fuse/overlap
collective-adjacent compute more aggressively than the current `auto` setting.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_enable_collective_compute_fusion=true`. Do not carry v076 resource
sharing, v077 Mosaic async collective fusion, v078/v079 multi-compute overlap,
or v080 dependent-collectives layer scheduling.

**Predicted signal**: Loss stays valid and throughput exceeds the confirmed
51,021 tok/s / 36.6% MFU band, or the profile shows lower
convolution-fusion/collective-permute time without custom-call regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v081-ccfusion`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v081-shard-acts-s8k-bs3-tokamax-splash-collective-compute-fusion`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_collective_compute_fusion=true` to the v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v081-ccfusion` completed cleanly on 2026-06-04 with
normal loss, but regressed severely.

- Worker 0: 42,538 tok/s, 30.5% MFU
- Worker 1: 42,555 tok/s, 30.5% MFU
- Final loss: 12.0510 on both workers
- Exit code: 0

This is far below the confirmed 51,021 tok/s / 36.6% MFU frontier band.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v081-shard-acts-s8k-bs3-tokamax-splash-collective-compute-fusion/2026_06_04_04_36_57`

Profile:

- MXU utilization: 47.4%
- Total time: 111,181.3 ms; idle 285.6 ms / 0.3%
- `jit_train_step`: 110,895.0 ms / 99.7%
- Convolution fusion: 62,001.6 ms / 55.8%
- Custom-call: 22,064.4 ms / 19.8%
  - `splash_mha_dkv_no_residuals.1`: 11,640.5 ms / 10.5%
  - `splash_mha_fwd_residuals.2/3`: 10,424.0 ms / 9.4%
- Loop fusion: 18,352.9 ms / 16.5%
- Data formatting: 3,886.3 ms / 3.5%
- Collective-permute-start: 1,003.6 ms / 0.9%
- Peak HBM: 28.34 GiB of 31.25 GiB; 2.91 GiB free; 0.05% fragmentation

The regression is primarily the convolution-fusion bucket: it rises from about
45.7 s in the frontier profile to 62.0 s, and MXU utilization falls from about
56.7-56.8% to 47.4%.

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v081-shard-acts-s8k-bs3-tokamax-splash-collective-compute-fusion/hlo/`.

- HLO object count: 1,619
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,254,662 bytes
- Flagfile confirms:
  - `--xla_tpu_enable_collective_compute_fusion='true'`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`
  - `--xla_latency_hiding_scheduler_resource_sharing=false`
  - `--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=false`
  - `--xla_tpu_enable_layer_scheduler_for_dependent_collectives=auto`
  - `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=auto`

## Verdict

Refuted. Forcing collective-compute fusion creates a real alternate HLO and
severely hurts matmul/convolution-fusion efficiency. Do not carry this flag
forward.
