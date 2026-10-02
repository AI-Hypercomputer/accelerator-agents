---
title: "Qwen3 JAX v128 - shard acts s8k bs4 tokamax splash collective matmul none no memory pressure tracking"
type: experiment
hypothesis: "Scheduler memory-pressure tracking may be redundant after the v127 simplified collective-matmul-none stack and could constrain schedule freedom."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, memory-pressure, all-experimental-scheduler, ablation, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v128 - shard acts s8k bs4 tokamax splash collective matmul none no memory pressure tracking

## Hypothesis under test

**Hypothesis**: The v127 stack shows the broad all-experimental scheduler bundle
can be disabled without changing the optimized train-step HLO/profile. The next
remaining scheduler constraint is memory-pressure tracking. Disabling it may
either preserve the same final HLO or expose a slightly better schedule for the
batch-4 collective-matmul-none shape.

**Mechanism**: Start from v127 and change only:

```text
--xla_tpu_enable_scheduler_memory_pressure_tracking=false
```

Keep the simplified v127 stack fixed:

```text
--xla_tpu_enable_all_experimental_scheduler_features=false
--xla_latency_hiding_scheduler_enable_selective_resources=true
--xla_tpu_enable_latency_hiding_layer_scheduler=false
--xla_tpu_enable_sparse_core_collective_aggregator=false
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: Clean completion in the 56.45k tok/s / 40.5% MFU band,
with either identical HLO or a small profile improvement in loop fusion or
all-reduce-scatter fusion.

**Falsification criterion**: Compile/runtime failure, invalid loss, HBM OOM, or
clean completion below the confirmed v123/v126/v127 band.

## Setup

GKE workload `alekseyv-qwen3-v128-bs4-nomemtrack`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v128-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-memory-pressure-tracking`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 stack with scheduler memory-pressure tracking disabled.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v128-bs4-nomemtrack` completed cleanly on both
workers.

- Worker0: average **56,448 tok/s** / 7,056 tok/s/chip / **40.5% MFU**;
  best observed step **56,592 tok/s**; final loss 12.0461.
- Worker1: average **56,428 tok/s** / 7,053 tok/s/chip / **40.5% MFU**;
  best observed step **56,597 tok/s**; final loss 12.0461.
- Profile window: steps 12-14 captured successfully.
- Correctness: no NaNs, normal loss trajectory, exit code 0 on both workers.

This is an in-band tie/slight regression relative to v127 and does not improve
the v123 best-observed-step frontier.

## Profile

XProf run:

```text
2026-06-04-qwen3-jax-v128-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-memory-pressure-tracking/2026_06_04_12_54_39
```

Profile summary:

- Step-time average: **1999.3 ms**
- MXU utilization: **64.1%**
- Total profile time: **111,850.3 ms**
- Idle: **282.5 ms / 0.3%**
- Peak HBM: **26.91 GiB** of 31.25 GiB

Top device buckets:

- Convolution fusion: **60,200.4 ms / 53.8%**
- Custom-call: **29,347.9 ms / 26.2%**
  - `splash_mha_dkv_no_residuals.1`: **15,477.5 ms / 13.8%**
  - `splash_mha_fwd_residuals.2/.3`: **13,870.4 ms / 12.4%**
- Loop fusion: **11,679.1 ms / 10.4%**

The train-step program hash remains unchanged:

```text
jit_train_step(16296087512277873197)
```

## HLO Dump

Fresh dumps were emitted under:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v128-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-memory-pressure-tracking/hlo/
```

The final train-step module is `module_0267.jit_train_step.cl_854318611`.

- Optimized HLO size: **787,978 bytes**
- SHA256: `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`
- Memory report: **28,862,302,652 bytes / 26.88 GiB**

This is byte-identical to v123/v126/v127. The flagfile confirms the intended
change was applied:

```text
--xla_tpu_enable_scheduler_memory_pressure_tracking='false'
--xla_sc_enable_scheduler_memory_pressure_tracking=true
--xla_tpu_enable_all_experimental_scheduler_features=false
--xla_latency_hiding_scheduler_enable_selective_resources='true'
--xla_tpu_enable_latency_hiding_layer_scheduler='false'
--xla_tpu_enable_sparse_core_collective_aggregator='false'
--xla_tpu_all_gather_collective_matmul_mode=NONE
--xla_tpu_reduce_scatter_collective_matmul_mode=NONE
```

The sparse-core memory-pressure flag remains true. That is acceptable for this
one-factor TPU scheduler ablation, but it also explains why the final program may
stay unchanged.

## Verdict

**Refuted / tie.** Disabling TPU scheduler memory-pressure tracking is valid but
does not improve throughput, HBM, or the optimized train-step program. Keep the
v127 simplified baseline with TPU memory-pressure tracking enabled for future
probes.

Next test: disable custom-call scoped VMEM adjustments from the v127 stack to
check whether the v072/v073 pre-collective-matmul-none gain is still needed after
the final batch-4 collective-matmul-none schedule.
