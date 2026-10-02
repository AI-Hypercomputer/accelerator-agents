---
title: "Qwen3 JAX v083 - shard acts s8k bs3 tokamax splash fuse kloop fusions"
type: experiment
hypothesis: "Allowing async collective fusion to fuse kloop fusions may improve overlap without the harmful collective-compute fusion path."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, async-collective-fusion, kloop-fusion, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, invalid, compile-failure, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v083 - shard acts s8k bs3 tokamax splash fuse kloop fusions

## Hypothesis under test

**Hypothesis**: The frontier flagfile leaves
`--xla_tpu_enable_async_collective_fusion_fuse_kloop_fusions=false`. Enabling it
may let async collective fusion include kloop fusion regions and improve overlap
without forcing the much broader and harmful collective-compute fusion path
tested in v081.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_enable_async_collective_fusion_fuse_kloop_fusions=true`. Do not carry
v076-v082 refuted/tie scheduler variants.

**Predicted signal**: Loss stays valid and throughput exceeds the confirmed
51,021 tok/s / 36.6% MFU band, or profile total time drops without increasing
convolution/custom-call time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v083-kloopacf`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v083-shard-acts-s8k-bs3-tokamax-splash-fuse-kloop-fusions`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_async_collective_fusion_fuse_kloop_fusions=true` to the
  v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v083-kloopacf` failed during train-step compile.

- Visible pod: `alekseyv-qwen3-v083-kloopacf-slice-job-0-0-pd9tc`
- Exit code: 139
- No train steps completed.
- Worker 1 pod was gone by the time the jobset ended; worker 0 retained the
  crash log.

Crash signature:

```text
F0604 04:57:31.991357 async_all_gather_emitter.cc:52] Check failed: Init( false) is OK (INVALID_ARGUMENT: Unable to find constant of type: 3
```

## Profile

No XProf run was produced because the workload failed before executing training
steps.

## HLO Dump

Partial HLO artifacts exist under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v083-shard-acts-s8k-bs3-tokamax-splash-fuse-kloop-fusions/hlo/`.

- HLO object count: 1,604
- Train-step after-optimization text exists:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,664,434 bytes
- Buffer assignment and memory reports were written before the libtpu fatal.
- Flagfile confirms:
  - `--xla_tpu_enable_async_collective_fusion_fuse_kloop_fusions=true`
  - `--xla_tpu_async_collective_fusion_fuse_multiple_collectives=false`
  - `--xla_tpu_enable_collective_compute_fusion=auto`
  - `--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=false`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`

## Verdict

Invalid. Forcing async collective fusion of kloop fusions crashes libtpu's
async all-gather emitter during compile. Do not carry this flag forward.
