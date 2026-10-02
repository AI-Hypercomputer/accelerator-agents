---
title: "Qwen3 JAX v084 - shard acts s8k bs3 tokamax splash async start-done only"
type: experiment
hypothesis: "Restricting async collective fusion to start/done pairs may reduce collective overhead without triggering the harmful broader fusion paths."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, async-collective-fusion, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v084 - shard acts s8k bs3 tokamax splash async start-done only

## Hypothesis under test

**Hypothesis**: The frontier flagfile leaves
`--xla_tpu_async_collective_fusion_with_start_done_only=false`. Enabling it may
constrain async collective fusion to start/done regions and improve the residual
collective-permute schedule without carrying the broader fusion modes that were
neutral, harmful, or invalid in v076-v083.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_async_collective_fusion_with_start_done_only=true`. Do not carry
v076 resource sharing, v077 Mosaic async collective fusion, v078/v079
multi-compute overlap, v080 dependent-collectives layer scheduling, v081
collective-compute fusion, v082 multiple-collective fusion, or v083 kloop
fusion.

**Predicted signal**: Loss stays valid and throughput exceeds the confirmed
51,021 tok/s / 36.6% MFU band, or profile total time drops without increasing
convolution/custom-call time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v084-startdone`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v084-shard-acts-s8k-bs3-tokamax-splash-async-start-done-only`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_async_collective_fusion_with_start_done_only=true` to the
  v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v084-startdone` completed cleanly on 2026-06-04
with normal loss, but regressed below the confirmed v074/v075 frontier band.

- Worker 0: 50,629 tok/s, 36.3% MFU
- Worker 1: 50,656 tok/s, 36.3% MFU
- Final loss: 12.0509 on both workers
- Exit code: 0

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v084-shard-acts-s8k-bs3-tokamax-splash-async-start-done-only/2026_06_04_05_11_46`

Profile:

- MXU utilization: 56.4%
- Total time: 93,503.4 ms; idle 298.6 ms / 0.3%
- `jit_train_step`: 93,204.1 ms / 99.7%
- Convolution fusion: 45,312.8 ms / 48.5%
- Custom-call: 22,075.7 ms / 23.6%
  - `splash_mha_dkv_no_residuals.1`: 11,644.5 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,431.2 ms / 11.2%
- Loop fusion: 16,592.7 ms / 17.7%
- Data formatting: 3,669.4 ms / 3.9%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v084-shard-acts-s8k-bs3-tokamax-splash-async-start-done-only/hlo/`.

- HLO object count: 1,619
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,215,911 bytes
- Flagfile confirms:
  - `--xla_tpu_async_collective_fusion_with_start_done_only=true`
  - `--xla_tpu_async_collective_fusion_fuse_multiple_collectives=false`
  - `--xla_tpu_enable_async_collective_fusion_fuse_kloop_fusions=false`
  - `--xla_tpu_enable_collective_compute_fusion=auto`
  - `--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=false`
  - `--xla_latency_hiding_scheduler_resource_sharing=false`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`

## Verdict

Refuted. Restricting async collective fusion to start/done pairs creates a real
schedule variant but loses roughly 365 tok/s versus the confirmed 51,021 tok/s
band. It does not reduce the custom-call bucket meaningfully and lowers MXU
utilization to 56.4%. Do not carry this flag forward.
