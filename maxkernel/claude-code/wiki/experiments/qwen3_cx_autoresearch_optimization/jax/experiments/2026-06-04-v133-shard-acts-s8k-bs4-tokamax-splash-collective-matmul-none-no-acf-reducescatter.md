---
title: "Qwen3 JAX v133 - shard acts s8k bs4 tokamax splash collective matmul none no ACF reduce-scatter"
type: experiment
hypothesis: "After collective-matmul modes are disabled, the reduce-scatter async-collective-fusion subtype may be redundant; revalidate it on the simplified v127 stack."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, async-collective-fusion, reduce-scatter, latency-hiding, scheduler, ablation, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v133 - shard acts s8k bs4 tokamax splash collective matmul none no ACF reduce-scatter

## Hypothesis under test

**Hypothesis**: v114 showed that disabling only the reduce-scatter async
collective fusion subtype was an in-band tie on the pre-collective-matmul-none
layeragg stack. Revalidate the same narrow subtype ablation after v116-v127
disabled both collective-matmul modes and simplified the final frontier HLO.

**Mechanism**: Start from v127 and change only:

```text
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
```

Keep base ACF enabled and keep all-gather plus multiple-step ACF enabled:

```text
--xla_tpu_enable_async_collective_fusion=true
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
--xla_tpu_enable_async_collective_fusion_multiple_steps=true
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

**Predicted signal**: If reduce-scatter ACF is redundant post-cmnone, the final
HLO/program should stay byte-identical to v127 and throughput should remain in
the 56.4k-56.6k tok/s band. If it is required, reduce-scatter or all-reduce
scatter cost should rise and throughput should regress.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with regression, or clean completion below the confirmed v123/v127 band.

## Setup

GKE workload `alekseyv-qwen3-v133-bs4-noacfrs`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v133-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-reducescatter`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 stack with only reduce-scatter ACF subtype disabled.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v133-bs4-noacfrs` completed successfully on both
workers.

- **Worker 0**: 56,440 tok/s avg over 18 measured steps, 7,055 tok/s/chip,
  40.5% MFU; best observed steady step 56,561 tok/s; final loss 12.0461.
- **Worker 1**: 56,414 tok/s avg over 18 measured steps, 7,052 tok/s/chip,
  40.4% MFU; best observed steady step 56,557 tok/s; final loss 12.0461.
- **Profile dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v133-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-reducescatter`

## Profile

XProf run:
`2026-06-04-qwen3-jax-v133-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-reducescatter/2026_06_04_13_56_07`

- **Step time average**: 2000.8 ms
- **MXU utilization**: 64.1%
- **Total device time**: 111,893.1 ms
- **Idle**: 327.2 ms / 0.3%
- **Peak HBM**: 26.91 GiB of 31.25 GiB
- **Train-step program**: `jit_train_step(16296087512277873197)`
- **Top buckets**:
  - convolution fusion: 60,209.4 ms / 53.8%
  - custom-call: 29,348.0 ms / 26.2%
  - `splash_mha_dkv_no_residuals.1`: 15,477.5 ms / 13.8%
  - forward residual Splash custom-calls: 13,870.5 ms / 12.4% combined
  - loop fusion: 11,678.4 ms / 10.4%
  - all-reduce-scatter fusion: 3,762.6 ms / 3.4%

This is in the normal v123/v127 profile band. There is no explicit
reduce-scatter regression, but there is also no throughput improvement over
v127 or the v123 best-observed frontier.

## HLO Dump

- **Train-step module**:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **Optimized HLO size**: 787,978 bytes
- **Optimized HLO SHA256**:
  `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`
- **Memory report**: 28,862,302,652 bytes / 26.88 GiB
- **Flagfile confirmation**:
  `--xla_tpu_enable_async_collective_fusion=true`,
  `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true`,
  `--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false`,
  `--xla_tpu_enable_async_collective_fusion_multiple_steps=true`,
  AG backward pipelining `true`, selective resources `true`, all-experimental
  scheduler bundle `false`, layer scheduler and sparse-core aggregator
  explicitly `false`, and both collective-matmul modes `NONE`.

The optimized train-step HLO and train-step program hash are byte-identical to
the v123/v126/v127 frontier shape.

## Verdict

Supported/tie for redundancy, not a new frontier. Disabling only the
reduce-scatter ACF subtype after collective-matmul-none preserves the exact
frontier train-step HLO/profile and stays in-band at 56.4k tok/s / 40.5% MFU,
but it does not improve the v127 average or the v123 best-observed steady step.

Do not treat this as a throughput improvement. Continue the subtype sweep from
the v127 one-mechanism baseline by disabling only ACF multiple-steps.
