---
title: "Qwen3 JAX v080 - shard acts s8k bs3 tokamax splash dependent collectives layer scheduler"
type: experiment
hypothesis: "Forcing the TPU layer scheduler for dependent collectives may improve overlap in the remaining collective-permute and all-gather-adjacent schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, scheduler, layer-scheduler, dependent-collectives, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, refuted, tie, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v080 - shard acts s8k bs3 tokamax splash dependent collectives layer scheduler

## Hypothesis under test

**Hypothesis**: The confirmed v074/v075 frontier still has about 1% collective
permute start time and many all-gather-adjacent fusion nodes inside the
convolution-fusion bucket. Forcing
`--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true` may let the
layer scheduler handle dependent collective chains more aggressively than the
current `auto` setting.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true`. Do not carry
v076 resource sharing, v077 Mosaic async collective fusion, or v078/v079
multi-compute overlap.

**Predicted signal**: Loss stays valid and throughput exceeds the confirmed
51,021 tok/s / 36.6% MFU band, or collective-permute/convolution-fusion profile
time improves without increasing custom-call or loop-fusion time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v080-depcoll`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v080-shard-acts-s8k-bs3-tokamax-splash-dependent-collectives-layer-scheduler`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true` to the
  v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v080-depcoll` completed cleanly on 2026-06-04 with
normal loss.

- Worker 0: 50,997 tok/s, 36.6% MFU
- Worker 1: 51,017 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This remains inside the frontier band but below the confirmed 51,021 tok/s
best.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v080-shard-acts-s8k-bs3-tokamax-splash-dependent-collectives-layer-scheduler/2026_06_04_04_25_41`

Profile:

- MXU utilization: 56.8%
- Total time: 92,798.2 ms; idle 263.2 ms / 0.3%
- `jit_train_step`: 92,534.3 ms / 99.7%
- Convolution fusion: 45,713.0 ms / 49.3%
- Custom-call: 22,057.2 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,641.1 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,416.1 ms / 11.2%
- Loop fusion: 16,726.0 ms / 18.0%
- Data formatting: 3,772.9 ms / 4.1%
- Collective-permute-start: 932.4 ms / 1.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v080-shard-acts-s8k-bs3-tokamax-splash-dependent-collectives-layer-scheduler/hlo/`.

- HLO object count: 1,619
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,375,271 bytes
- Flagfile confirms:
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`
  - `--xla_tpu_enable_layer_scheduler_for_dependent_collectives='true'`
  - `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=auto`
  - `--xla_latency_hiding_scheduler_resource_sharing=false`
  - `--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=false`

## Verdict

Refuted/tie. The dependent-collectives layer-scheduler flag is valid and
in-band, but it does not beat the confirmed frontier and slightly increases
collective-permute/data-formatting profile time. Do not carry it forward.
