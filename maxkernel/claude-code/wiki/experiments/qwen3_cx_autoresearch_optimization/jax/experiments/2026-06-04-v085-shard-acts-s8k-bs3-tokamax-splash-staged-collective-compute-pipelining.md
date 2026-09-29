---
title: "Qwen3 JAX v085 - shard acts s8k bs3 tokamax splash staged collective-compute pipelining"
type: experiment
hypothesis: "Staged collective-compute pipelining may improve overlap without forcing the harmful collective-compute fusion path."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-pipelining, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, invalid, compile-failure, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v085 - shard acts s8k bs3 tokamax splash staged collective-compute pipelining

## Hypothesis under test

**Hypothesis**: v081 showed that forcing full collective-compute fusion is
harmful, but the frontier flagfile still leaves
`--xla_tpu_enable_staged_collective_compute_pipelining=auto`. Explicit staged
collective-compute pipelining may improve overlap around collectives without
collapsing the graph into the convolution-fusion-heavy schedule seen in v081.

**Mechanism**: Keep the v074/v075 frontier fixed and add only
`--xla_tpu_enable_staged_collective_compute_pipelining=true`. Do not carry
v076-v084 refuted/tie scheduler variants.

**Predicted signal**: Loss stays valid and throughput exceeds the confirmed
51,021 tok/s / 36.6% MFU band, or total profile time drops without increasing
convolution/custom-call time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v074/v075 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v085-stagedcc`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v085-shard-acts-s8k-bs3-tokamax-splash-staged-collective-compute-pipelining`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_staged_collective_compute_pipelining=true` to the
  v074/v075 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v085-stagedcc` failed during train-step compile before
any measured step.

- Retained pod: `alekseyv-qwen3-v085-stagedcc-slice-job-0-1-dz4lt`
- Exit code: 139
- No train steps completed.

Crash signature:

```text
*** SIGSEGV (@0x88), si_code=1
xla::jellyfish::fusion_util::IsDynamicUpdateSliceAligned()
xla::jellyfish::collective_compute_pipeliner_util::GetStagedAllGatherDotInfo()
xla::jellyfish::CollectiveComputeDecomposer::DecomposeAllGatherEinsums()
xla::jellyfish::CollectiveComputeDecomposer::RunImpl()
```

## Profile

No XProf run was produced because the workload failed before executing training
steps.

## HLO Dump

Partial HLO artifacts exist under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v085-shard-acts-s8k-bs3-tokamax-splash-staged-collective-compute-pipelining/hlo/`.

- HLO object count: 1,600
- Train-step before-optimization text exists:
  - `module_0267.jit_train_step.cl_854318611.before_optimizations.txt`
- No train-step after-optimization text was written before the crash.
- Flagfile confirms:
  - `--xla_tpu_enable_staged_collective_compute_pipelining='true'`
  - `--xla_tpu_enable_collective_compute_fusion=auto`
  - `--xla_tpu_async_collective_fusion_with_start_done_only=false`
  - `--xla_tpu_async_collective_fusion_fuse_multiple_collectives=false`
  - `--xla_tpu_enable_async_collective_fusion_fuse_kloop_fusions=false`
  - `--xla_latency_hiding_scheduler_resource_sharing=false`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`

## Verdict

Invalid. Forcing staged collective-compute pipelining crashes libtpu's
collective compute decomposer while analyzing staged all-gather/dot structure.
Do not carry this flag forward.
