---
title: "Qwen3 JAX v082 - shard acts s8k bs3 tokamax splash fuse multiple collectives"
type: experiment
hypothesis: "Fusing multiple async collectives may reduce the persistent collective-permute overhead without the harmful collective-compute fusion transformation."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported/tie
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, async-collective-fusion, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, supported, tie, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v082 - shard acts s8k bs3 tokamax splash fuse multiple collectives

## Hypothesis under test

**Hypothesis**: v081 shows full collective-compute fusion is harmful, but the
frontier flagfile still has
`--xla_tpu_async_collective_fusion_fuse_multiple_collectives=false` while async
collective fusion itself is enabled. Fusing only multiple async collectives may
reduce the persistent collective-permute overhead without changing the
matmul/custom-call schedule as aggressively as v081.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_async_collective_fusion_fuse_multiple_collectives=true`. Do not carry
v076 resource sharing, v077 Mosaic async collective fusion, v078/v079
multi-compute overlap, v080 dependent-collectives layer scheduling, or v081
collective-compute fusion.

**Predicted signal**: Loss stays valid and throughput exceeds the confirmed
51,021 tok/s / 36.6% MFU band, or collective-permute/data-formatting time drops
without increasing convolution/custom-call time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v082-fusemulti`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v082-shard-acts-s8k-bs3-tokamax-splash-fuse-multiple-collectives`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_async_collective_fusion_fuse_multiple_collectives=true` to the
  v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v082-fusemulti` completed cleanly on 2026-06-04
with normal loss.

- Worker 0: 50,995 tok/s, 36.6% MFU
- Worker 1: 51,021 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This ties the confirmed best worker value but does not improve the durable
frontier.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v082-shard-acts-s8k-bs3-tokamax-splash-fuse-multiple-collectives/2026_06_04_04_47_42`

Profile:

- MXU utilization: 56.8%
- Total time: 92,794.1 ms; idle 300.6 ms / 0.3%
- `jit_train_step`: 92,492.9 ms / 99.7%
- Convolution fusion: 45,726.2 ms / 49.3%
- Custom-call: 22,056.2 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,641.2 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,415.0 ms / 11.2%
- Loop fusion: 16,719.1 ms / 18.0%
- Data formatting: 3,769.4 ms / 4.1%
- Collective-permute-start: 884.1 ms / 1.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v082-shard-acts-s8k-bs3-tokamax-splash-fuse-multiple-collectives/hlo/`.

- HLO object count: 2,218
- Train-step after-optimization dumps:
  - `module_0264.jit_train_step.cl_854318611.after_optimizations.txt`
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,375,271 bytes
- Flagfile confirms:
  - `--xla_tpu_async_collective_fusion_fuse_multiple_collectives=true`
  - `--xla_tpu_enable_collective_compute_fusion=auto`
  - `--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=false`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`

## Verdict

Supported/tie. Fusing multiple async collectives is valid and stays in the
frontier band, but it does not beat the confirmed 51,021 tok/s result. The
profile is essentially unchanged; do not treat it as a durable improvement.
