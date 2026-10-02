---
title: "Qwen3 JAX v077 - shard acts s8k bs3 tokamax splash mosaic async collective fusion"
type: experiment
hypothesis: "Allowing async collective fusion across Mosaic custom calls may improve overlap around the remaining Splash custom-call bucket."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, scheduler, async-collective-fusion, mosaic, custom-call, selective-resources, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v077 - shard acts s8k bs3 tokamax splash mosaic async collective fusion

## Hypothesis under test

**Hypothesis**: v074/v075 confirmed a selective-resources scheduler frontier,
while v076 showed resource sharing is only a tie/slight regression. The
frontier profile still spends about 23.8% in Mosaic Splash custom calls and has
small but persistent collective-permute/data-formatting work. Enabling
`--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true` may
allow async collective fusion to schedule across Mosaic custom-call boundaries
and recover overlap.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true`.
Do not carry v076's resource-sharing flag.

**Predicted signal**: Loss stays valid and throughput exceeds the v074/v075
band, ideally above 51,021 tok/s / 36.6% MFU, or profile total time drops with
collective-permute/data-formatting below the v074-v076 band.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below the v074/v075 throughput band
without a profile improvement.

## Setup

Planned GKE workload `alekseyv-qwen3-v077-mosaicacf`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v077-shard-acts-s8k-bs3-tokamax-splash-mosaic-async-collective-fusion`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true` to
  the v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v077-mosaicacf` completed cleanly on 2026-06-04
with normal loss.

- Worker 0: 50,976 tok/s, 36.5% MFU
- Worker 1: 50,952 tok/s, 36.5% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is below the confirmed v074/v075 frontier of 51,021 tok/s / 36.6% MFU.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v077-shard-acts-s8k-bs3-tokamax-splash-mosaic-async-collective-fusion/2026_06_04_03_57_07`

Profile:

- MXU utilization: 56.8%
- Total time: 92,857.2 ms; idle 272.7 ms / 0.3%
- `jit_train_step`: 92,583.8 ms / 99.7%
- Convolution fusion: 45,743.5 ms / 49.3%
- Custom-call: 16,859.2 ms / 18.2%
  - `splash_mha_dkv_no_residuals.1`: 11,647.3 ms / 12.5%
  - `splash_mha_fwd_residuals.3`: 5,211.8 ms / 5.6%
- Custom fusion: 5,839.1 ms / 6.3%
- Loop fusion: 16,739.4 ms / 18.0%
- Data formatting: 3,760.0 ms / 4.0%
- Collective-permute-start: 887.9 ms / 1.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

The flag moved work out of the `custom-call` bucket into `custom fusion`, but
the combined custom-call/custom-fusion share is still about 24.5% and wall time
regressed.

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v077-shard-acts-s8k-bs3-tokamax-splash-mosaic-async-collective-fusion/hlo/`.

- HLO object count: 1,611
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,384,405 bytes
- Flagfile confirms:
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`
  - `--xla_latency_hiding_scheduler_resource_sharing=false`
  - `--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true`
  - `--xla_latency_hiding_scheduler_rerun=3`
  - `--xla_tpu_enable_ag_backward_pipelining=true`

## Verdict

Refuted. The Mosaic custom-call async collective fusion flag is valid and
active, but it worsens throughput and simply reclassifies part of the custom
call path as custom fusion. Do not carry it forward.
