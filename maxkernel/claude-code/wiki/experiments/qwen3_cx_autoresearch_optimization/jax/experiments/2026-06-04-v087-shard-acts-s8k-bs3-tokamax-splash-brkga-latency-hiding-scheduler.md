---
title: "Qwen3 JAX v087 - shard acts s8k bs3 tokamax splash BRKGA latency-hiding scheduler"
type: experiment
hypothesis: "Forcing the BRKGA latency-hiding scheduler may find a better schedule than the auto-selected scheduler at the selective-resources frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported/tie
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, latency-hiding, brkga-scheduler, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, supported, tie, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v087 - shard acts s8k bs3 tokamax splash BRKGA latency-hiding scheduler

## Hypothesis under test

**Hypothesis**: v086 shows forcing ILP LHS is valid but not better. The
frontier flagfile still leaves
`--xla_tpu_enable_brkga_latency_hiding_scheduler=auto`. Forcing BRKGA may search
a different latency-hiding schedule and improve the 51,021 tok/s band without
using the collective-fusion paths that regressed or crashed.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_enable_brkga_latency_hiding_scheduler=true`. Do not carry v076-v086
refuted/tie scheduler variants.

**Predicted signal**: Loss stays valid and throughput exceeds the confirmed
51,021 tok/s / 36.6% MFU band, or total profile time drops with no increase in
custom-call or data-formatting time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v087-brkgalhs`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v087-shard-acts-s8k-bs3-tokamax-splash-brkga-latency-hiding-scheduler`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_brkga_latency_hiding_scheduler=true` to the v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v087-brkgalhs` completed cleanly on 2026-06-04
with normal loss.

- Worker 0: 51,023 tok/s, 36.6% MFU
- Worker 1: 50,997 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is a tiny best-worker high-water mark relative to the confirmed 51,021
tok/s band, but it does not exceed the existing best-observed 51,025 tok/s from
v078 and has no clear profile shift.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v087-shard-acts-s8k-bs3-tokamax-splash-brkga-latency-hiding-scheduler/2026_06_04_05_42_06`

Profile:

- MXU utilization: 56.8%
- Total time: 92,790.1 ms; idle 282.5 ms / 0.3%
- `jit_train_step`: 92,507.0 ms / 99.7%
- Convolution fusion: 45,721.6 ms / 49.3%
- Custom-call: 22,056.6 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,641.0 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,415.6 ms / 11.2%
- Loop fusion: 16,700.8 ms / 18.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v087-shard-acts-s8k-bs3-tokamax-splash-brkga-latency-hiding-scheduler/hlo/`.

- HLO object count: 1,655
- Train-step after-optimization dumps:
  - `module_0262.jit_train_step.cl_854318611.after_optimizations.txt`
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,375,271 bytes
- Flagfile confirms:
  - `--xla_tpu_enable_brkga_latency_hiding_scheduler='true'`
  - `--xla_tpu_enable_ilp_latency_hiding_scheduler=auto`
  - `--xla_tpu_enable_staged_collective_compute_pipelining=auto`
  - `--xla_tpu_enable_collective_compute_fusion=auto`
  - `--xla_tpu_async_collective_fusion_with_start_done_only=false`
  - `--xla_tpu_async_collective_fusion_fuse_multiple_collectives=false`
  - `--xla_tpu_enable_async_collective_fusion_fuse_kloop_fusions=false`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`

## Verdict

Supported/tie. Forcing BRKGA LHS is valid and produces a tiny best-worker
51,023 tok/s result, but the paired worker is lower, the profile is effectively
unchanged, and the run does not exceed the existing 51,025 tok/s best-observed
single-worker mark from v078. Do not treat it as a durable new frontier without
a rerun.
