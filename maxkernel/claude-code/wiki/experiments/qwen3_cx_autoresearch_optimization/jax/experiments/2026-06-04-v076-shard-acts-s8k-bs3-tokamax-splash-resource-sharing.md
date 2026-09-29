---
title: "Qwen3 JAX v076 - shard acts s8k bs3 tokamax splash resource sharing"
type: experiment
hypothesis: "Latency-hiding scheduler resource sharing may further improve the confirmed selective-resources scheduler frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, scheduler, resource-sharing, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v076 - shard acts s8k bs3 tokamax splash resource sharing

## Hypothesis under test

**Hypothesis**: v074/v075 confirmed that enabling selective resources improves
latency-hiding scheduler placement. The flagfile still leaves
`--xla_latency_hiding_scheduler_resource_sharing=false`; enabling resource
sharing may allow the scheduler to use shared-resource accounting more
aggressively and further improve the 51k tok/s frontier.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_latency_hiding_scheduler_resource_sharing=true`.

**Predicted signal**: Loss stays valid and throughput exceeds the v074/v075
band, ideally above 51,021 tok/s / 36.6% MFU or with lower profile total time.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below the v074/v075 band.

## Setup

Planned GKE workload `alekseyv-qwen3-v076-sharing`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v076-shard-acts-s8k-bs3-tokamax-splash-resource-sharing`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_latency_hiding_scheduler_resource_sharing=true` to the v074/v075
  bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v076-sharing` completed cleanly on 2026-06-04 with
normal loss.

- Worker 0: 50,980 tok/s, 36.6% MFU
- Worker 1: 51,017 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is just below the confirmed v074/v075 best worker value of 51,021 tok/s,
so resource sharing does not supersede the selective-resources frontier.

## Profile

XProf runs:

- `2026-06-04-qwen3-jax-v076-shard-acts-s8k-bs3-tokamax-splash-resource-sharing/2026_06_04_03_44_03`
- `2026-06-04-qwen3-jax-v076-shard-acts-s8k-bs3-tokamax-splash-resource-sharing/2026_06_04_03_44_04`

Primary host (`03_44_04`) profile:

- MXU utilization: 56.7%
- Total time: 46,342.9 ms; idle 83.4 ms / 0.2%
- `jit_train_step`: 46,259.1 ms / 99.8%
- Convolution fusion: 22,847.2 ms / 49.3%
- Custom-call: 11,026.3 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 5,820.3 ms / 12.6%
  - `splash_mha_fwd_residuals.2/3`: 5,206.0 ms / 11.2%
- Loop fusion: 8,345.0 ms / 18.0%
- Data formatting: 1,881.4 ms / 4.1%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

The second host agrees: MXU 56.7%, peak HBM 28.32 GiB, custom-call 23.7%,
and loop fusion 18.0%.

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v076-shard-acts-s8k-bs3-tokamax-splash-resource-sharing/hlo/`.

- HLO object count: 2,226
- Train-step after-optimization dumps:
  - `module_0264.jit_train_step.cl_854318611.after_optimizations.txt`
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,375,271 bytes
- Flagfile confirms:
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`
  - `--xla_latency_hiding_scheduler_resource_sharing=true`
  - `--xla_latency_hiding_scheduler_rerun=3`
  - `--xla_tpu_enable_ag_backward_pipelining=true`

The changed flag produced a real HLO dump, so this was not merely a persistent
cache replay.

## Verdict

Refuted. Resource sharing is valid and does not hurt badly, but it is a
tie/slight regression relative to the confirmed v074/v075 selective-resources
frontier. Keep selective resources enabled; do not carry
`--xla_latency_hiding_scheduler_resource_sharing=true` forward by default.
