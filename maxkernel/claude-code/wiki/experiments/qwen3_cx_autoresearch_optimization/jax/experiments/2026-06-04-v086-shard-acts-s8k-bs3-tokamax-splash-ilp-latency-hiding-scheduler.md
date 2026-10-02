---
title: "Qwen3 JAX v086 - shard acts s8k bs3 tokamax splash ILP latency-hiding scheduler"
type: experiment
hypothesis: "Forcing the ILP latency-hiding scheduler may find a better schedule than the auto-selected scheduler at the selective-resources frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported/tie
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, latency-hiding, ilp-scheduler, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, supported, tie, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v086 - shard acts s8k bs3 tokamax splash ILP latency-hiding scheduler

## Hypothesis under test

**Hypothesis**: The v074/v075 frontier relies on the latency-hiding scheduler
with selective resources, but leaves
`--xla_tpu_enable_ilp_latency_hiding_scheduler=auto`. Forcing the ILP scheduler
may produce a slightly better schedule without engaging the collective fusion
paths that regressed or crashed in v081-v085.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_enable_ilp_latency_hiding_scheduler=true`. Do not carry v076-v085
refuted/tie scheduler variants.

**Predicted signal**: Loss stays valid and throughput exceeds the confirmed
51,021 tok/s / 36.6% MFU band, or total profile time drops with no increase in
custom-call or data-formatting time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v086-ilplhs`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v086-shard-acts-s8k-bs3-tokamax-splash-ilp-latency-hiding-scheduler`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_ilp_latency_hiding_scheduler=true` to the v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v086-ilplhs` completed cleanly on 2026-06-04
with normal loss.

- Worker 0: 51,016 tok/s, 36.6% MFU
- Worker 1: 50,989 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is in the frontier band but remains below the confirmed 51,021 tok/s
worker value from v074/v075.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v086-shard-acts-s8k-bs3-tokamax-splash-ilp-latency-hiding-scheduler/2026_06_04_05_31_51`

Profile:

- MXU utilization: 56.8%
- Total time: 92,818.8 ms; idle 288.9 ms / 0.3%
- `jit_train_step`: 92,529.2 ms / 99.7%
- Convolution fusion: 45,704.1 ms / 49.2%
- Custom-call: 22,057.4 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,641.1 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,416.2 ms / 11.2%
- Loop fusion: 16,702.6 ms / 18.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v086-shard-acts-s8k-bs3-tokamax-splash-ilp-latency-hiding-scheduler/hlo/`.

- HLO object count: 1,619
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,375,271 bytes
- Flagfile confirms:
  - `--xla_tpu_enable_ilp_latency_hiding_scheduler='true'`
  - `--xla_tpu_enable_brkga_latency_hiding_scheduler=auto`
  - `--xla_tpu_enable_staged_collective_compute_pipelining=auto`
  - `--xla_tpu_enable_collective_compute_fusion=auto`
  - `--xla_tpu_async_collective_fusion_with_start_done_only=false`
  - `--xla_tpu_async_collective_fusion_fuse_multiple_collectives=false`
  - `--xla_tpu_enable_async_collective_fusion_fuse_kloop_fusions=false`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`

## Verdict

Supported/tie. Forcing the ILP latency-hiding scheduler is valid and stays in
the frontier band, but it does not beat the confirmed 51,021 tok/s result and
does not produce a meaningful profile improvement. Do not treat it as a new
durable frontier.
