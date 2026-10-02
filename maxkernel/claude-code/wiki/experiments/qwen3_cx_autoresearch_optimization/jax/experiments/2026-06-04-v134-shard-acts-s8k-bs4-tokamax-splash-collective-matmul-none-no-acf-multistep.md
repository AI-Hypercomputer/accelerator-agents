---
title: "Qwen3 JAX v134 - shard acts s8k bs4 tokamax splash collective matmul none no ACF multiple-steps"
type: experiment
hypothesis: "After collective-matmul modes are disabled, async-collective-fusion multiple-steps may be redundant; revalidate it on the simplified v127 stack."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, async-collective-fusion, multiple-steps, latency-hiding, scheduler, ablation, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v134 - shard acts s8k bs4 tokamax splash collective matmul none no ACF multiple-steps

## Hypothesis under test

**Hypothesis**: v115 showed disabling async collective fusion multiple-steps
changed the pre-collective-matmul-none layeragg HLO but did not improve
throughput. Revalidate this final ACF subtype after v116-v127 disabled both
collective-matmul modes and simplified the frontier HLO.

**Mechanism**: Start from v127 and change only:

```text
--xla_tpu_enable_async_collective_fusion_multiple_steps=false
```

Keep base ACF enabled and keep all-gather plus reduce-scatter ACF enabled:

```text
--xla_tpu_enable_async_collective_fusion=true
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true
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

**Predicted signal**: If multiple-step ACF is still important post-cmnone, the
final HLO/program should change and throughput should regress. If it is now
redundant, the final HLO may stay byte-identical with throughput in the
56.4k-56.6k tok/s band.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with regression, or clean completion below the confirmed v123/v127 band.

## Setup

GKE workload `alekseyv-qwen3-v134-bs4-noacfms`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v134-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-multistep`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 stack with only ACF multiple-steps disabled.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v134-bs4-noacfms` completed cleanly on both
workers with normal loss ending at 12.0461.

- **Worker 0**: 56,294 tok/s avg, 7,037 tok/s/chip, 40.4% MFU; best steady
  step 56,425 tok/s.
- **Worker 1**: 56,269 tok/s avg, 7,034 tok/s/chip, 40.3% MFU; best steady
  step 56,424 tok/s.
- **Profile dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v134-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-multistep`

This is below the v127 best-worker average (56,453 tok/s) and below the v123
best-observed steady step (56,599 tok/s).

## Profile

- **XProf run**:
  `2026-06-04-qwen3-jax-v134-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-multistep/2026_06_04_14_08_09`
- **Step time avg**: 2005.7 ms
- **MXU utilization**: 64.0%
- **Total time**: 112,174.7 ms
- **Idle**: 316.4 ms / 0.3%
- **Peak HBM**: 26.91 GiB
- **Train-step program**: `jit_train_step(14153491495991100093)`

Top train-step buckets:

- `convolution fusion`: 61,031.5 ms / 54.4%
- `custom-call`: 29,451.2 ms / 26.3%
  - `splash_mha_dkv_no_residuals.1`: 15,490.0 ms / 13.8%
  - `splash_mha_fwd_residuals.*`: 13,961.2 ms / 12.4%
- `loop fusion`: 11,695.4 ms / 10.4%
- `all-reduce-scatter fusion`: 3,760.1 ms / 3.4%

## HLO Dump

- **HLO prefix**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v134-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-multistep/hlo/`
- **Optimized train-step**:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **HLO bytes**: 641,277
- **HLO SHA256**:
  `8a67b912870aa4546fc59e132fe9add8fa85086d4eea9e8b2c3d360ccf13dfcb`
- **Memory report**: 28,857,805,756 bytes / 26.88 GiB

The flagfile confirms `--xla_tpu_enable_async_collective_fusion_multiple_steps=false`
while base ACF, all-gather ACF, reduce-scatter ACF, selective resources,
memory-pressure tracking, custom-call scoped VMEM adjustment, layeragg false,
and both collective-matmul modes `NONE` were held fixed.

## Verdict

**Refuted.** Disabling async collective fusion multiple-steps after
collective-matmul-none selects a different train-step program/HLO and regresses
below the confirmed v123/v127 frontier band. Keep
`--xla_tpu_enable_async_collective_fusion_multiple_steps=true`.

The post-cmnone ACF ablation sweep is closed: base ACF and all-gather ACF are
required; reduce-scatter subtype is HLO-equivalent but not a gain; multiple-step
ACF is required for the best compiled program.
