---
title: "Qwen3 JAX v078 - shard acts s8k bs3 tokamax splash layer multicompute overlap"
type: experiment
hypothesis: "Forcing multi-compute overlap in the TPU latency-hiding layer scheduler may improve overlap around the remaining matmul/custom-call mix."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported/tie
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, scheduler, layer-scheduler, multi-compute-overlap, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, supported, tie, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v078 - shard acts s8k bs3 tokamax splash layer multicompute overlap

## Hypothesis under test

**Hypothesis**: v074/v075 established that latency-hiding scheduler selective
resources improve the target-sequence frontier, while v076 resource sharing and
v077 Mosaic async collective fusion did not help. The train-step profile still
has a large convolution-fusion bucket and about 23.8% Splash custom-call time.
Forcing `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true` may
allow the layer scheduler to overlap independent compute regions more
aggressively than the `auto` setting.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true`. Do not carry
v076 resource sharing or v077 Mosaic async collective fusion.

**Predicted signal**: Loss stays valid and throughput exceeds 51,021 tok/s /
36.6% MFU, or XProf train-step total time drops with convolution/custom-call
time better overlapped.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v078-multicompute`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v078-shard-acts-s8k-bs3-tokamax-splash-layer-multicompute-overlap`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true` to the
  v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v078-multicompute` completed cleanly on 2026-06-04
with normal loss.

- Worker 0: 51,025 tok/s, 36.6% MFU
- Worker 1: 51,001 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

Worker 0 is 4 tok/s above the confirmed v074/v075 best of 51,021 tok/s, but
worker 1 is below it. The exact v079 rerun landed at 51,021 / 51,009 tok/s, so
this is best interpreted as a supported tie rather than a durable new frontier.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v078-shard-acts-s8k-bs3-tokamax-splash-layer-multicompute-overlap/2026_06_04_04_08_02`

Profile:

- MXU utilization: 56.8%
- Total time: 92,767.2 ms; idle 283.2 ms / 0.3%
- `jit_train_step`: 92,483.2 ms / 99.7%
- Convolution fusion: 45,704.3 ms / 49.3%
- Custom-call: 22,056.7 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,641.2 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,415.5 ms / 11.2%
- Loop fusion: 16,709.4 ms / 18.0%
- Data formatting: 3,765.4 ms / 4.1%
- Collective-permute-start: 917.3 ms / 1.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v078-shard-acts-s8k-bs3-tokamax-splash-layer-multicompute-overlap/hlo/`.

- HLO object count: 1,615
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,375,271 bytes
- Flagfile confirms:
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`
  - `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler='true'`
  - `--xla_latency_hiding_scheduler_resource_sharing=false`
  - `--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=false`
  - `--xla_latency_hiding_scheduler_rerun=3`
  - `--xla_tpu_enable_ag_backward_pipelining=true`

## Verdict

Supported/tie. The run is valid and produces the best single-worker throughput
observed so far, but the exact v079 rerun did not reproduce the tiny gain.
Multi-compute overlap is non-harmful and in-band, but not a durable frontier
move.
