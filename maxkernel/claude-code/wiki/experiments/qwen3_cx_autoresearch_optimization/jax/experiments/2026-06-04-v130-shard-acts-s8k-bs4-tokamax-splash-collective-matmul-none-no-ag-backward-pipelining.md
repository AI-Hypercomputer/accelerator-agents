---
title: "Qwen3 JAX v130 - shard acts s8k bs4 tokamax splash collective matmul none no ag backward pipelining"
type: experiment
hypothesis: "AG backward pipelining may be redundant after the simplified v127 collective-matmul-none stack."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, ag-backward-pipelining, latency-hiding, scheduler, ablation, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v130 - shard acts s8k bs4 tokamax splash collective matmul none no ag backward pipelining

## Hypothesis under test

**Hypothesis**: `--xla_tpu_enable_ag_backward_pipelining=true` was part of the
pre-collective-matmul-none scheduler stack. With collective-matmul modes disabled
and the v127 simplified final HLO stable across several scheduler ablations, AG
backward pipelining may now be redundant or slightly harmful.

**Mechanism**: Start from v127 and change only:

```text
--xla_tpu_enable_ag_backward_pipelining=false
```

Keep the v127 simplified stack fixed:

```text
--xla_tpu_enable_all_experimental_scheduler_features=false
--xla_tpu_enable_scheduler_memory_pressure_tracking=true
--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true
--xla_latency_hiding_scheduler_enable_selective_resources=true
--xla_tpu_enable_latency_hiding_layer_scheduler=false
--xla_tpu_enable_sparse_core_collective_aggregator=false
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: Clean completion with either the exact same HLO/profile or
a small communication-profile shift. Prior pre-cmnone evidence suggests this is
more likely a tie/regression than a gain.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with regression, or clean completion below the confirmed v123/v127 band.

## Setup

GKE workload `alekseyv-qwen3-v130-bs4-noagpipe`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v130-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-ag-backward-pipelining`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 stack with AG backward pipelining disabled.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v130-bs4-noagpipe` completed successfully on both
workers.

- **Worker 0**: 56,449 tok/s avg over 18 measured steps, 7,056 tok/s/chip,
  40.5% MFU; best observed steady step 56,570 tok/s; final loss 12.0461.
- **Worker 1**: 56,431 tok/s avg over 18 measured steps, 7,054 tok/s/chip,
  40.5% MFU; best observed steady step 56,571 tok/s; final loss 12.0461.
- **Profile dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v130-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-ag-backward-pipelining`

## Profile

XProf run:
`2026-06-04-qwen3-jax-v130-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-ag-backward-pipelining/2026_06_04_13_18_53`

- **Step time average**: 1999.5 ms
- **MXU utilization**: 64.1%
- **Total device time**: 111,852.9 ms
- **Idle**: 300.2 ms / 0.3%
- **Peak HBM**: 26.91 GiB of 31.25 GiB
- **Train-step program**: `jit_train_step(16296087512277873197)`
- **Top buckets**:
  - convolution fusion: 60,205.7 ms / 53.8%
  - custom-call: 29,348.2 ms / 26.2%
  - `splash_mha_dkv_no_residuals.1`: 15,477.6 ms / 13.8%
  - `splash_mha_fwd_residuals.2` plus duplicate: 13,870.6 ms / 12.4%
  - loop fusion: 11,673.7 ms / 10.4%

## HLO Dump

- **Train-step module**:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **Optimized HLO size**: 787,978 bytes
- **Optimized HLO SHA256**:
  `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`
- **Memory report**: 28,862,302,652 bytes / 26.88 GiB
- **Flagfile confirmation**:
  `--xla_tpu_enable_ag_backward_pipelining=false`,
  `--xla_tpu_enable_all_experimental_scheduler_features=false`,
  `--xla_tpu_enable_scheduler_memory_pressure_tracking=true`,
  `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`,
  `--xla_latency_hiding_scheduler_enable_selective_resources=true`,
  layer scheduler and sparse-core aggregator explicitly `false`, and both
  collective-matmul modes `NONE`.

The optimized train-step HLO hash, size, memory report, and XProf program hash
are identical to the v123/v126/v127/v128/v129 final program.

## Verdict

Refuted/tie. Disabling AG backward pipelining is valid and leaves the final
compiled train-step program byte-identical, but it does not improve throughput:
worker averages are slightly below v127's 56,453 tok/s simplified-stack average
and best observed steps are below the v123 56,599 tok/s observed frontier. Do
not carry this ablation; keep the v127 simplified stack as the next baseline.

Next probe: revalidate async collective fusion on the post-collective-matmul-none
stack by disabling base `--xla_tpu_enable_async_collective_fusion` from v127.
