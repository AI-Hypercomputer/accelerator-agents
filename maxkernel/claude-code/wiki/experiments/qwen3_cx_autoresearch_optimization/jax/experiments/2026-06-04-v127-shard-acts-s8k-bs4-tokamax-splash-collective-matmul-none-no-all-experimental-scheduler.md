---
title: "Qwen3 JAX v127 - shard acts s8k bs4 tokamax splash collective matmul none no all experimental scheduler"
type: experiment
hypothesis: "The broad all-experimental scheduler bundle may be redundant after collective-matmul-none and the v126 simplified scheduler baseline."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, all-experimental-scheduler, ablation, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v127 - shard acts s8k bs4 tokamax splash collective matmul none no all experimental scheduler

## Hypothesis under test

**Hypothesis**: The broad all-experimental scheduler bundle was part of the
pre-v116 scheduler stack, but the v126 simplified collective-matmul-none
frontier may no longer need it. Disabling it may preserve the exact train-step
HLO or simplify scheduling without harming throughput.

**Mechanism**: Start from the v126 simplified stack and change only:

```text
--xla_tpu_enable_all_experimental_scheduler_features=false
```

Keep selective resources enabled, layeragg explicitly false, and collective
matmul modes disabled:

```text
--xla_latency_hiding_scheduler_enable_selective_resources=true
--xla_tpu_enable_latency_hiding_layer_scheduler=false
--xla_tpu_enable_sparse_core_collective_aggregator=false
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: Clean completion with the same 56.44k tok/s / 40.5% MFU
band and either identical HLO or a non-regressive simpler schedule.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with throughput regression, or clean completion below the confirmed v123/v126
band.

## Setup

GKE workload `alekseyv-qwen3-v127-bs4-noallsched`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v127-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-all-experimental-scheduler`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v126 stack with the all-experimental scheduler bundle disabled.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v127-bs4-noallsched` completed cleanly on both
workers.

- Worker0: average **56,453 tok/s** / 7,057 tok/s/chip / **40.5% MFU**;
  best observed step **56,573 tok/s**; final loss 12.0461.
- Worker1: average **56,437 tok/s** / 7,055 tok/s/chip / **40.5% MFU**;
  best observed step **56,579 tok/s**; final loss 12.0461.
- Profile window: steps 12-14 captured successfully.
- Correctness: no NaNs, normal loss trajectory, exit code 0 on both workers.

This is an in-band tie with the v123/v126 batch-4 frontier. It slightly raises
the worker0 run-average relative to v123/v126, but the best observed step is
below v123's 56,599 tok/s and the optimized train-step program is byte-identical
to v123/v126.

## Profile

XProf run:

```text
2026-06-04-qwen3-jax-v127-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-all-experimental-scheduler/2026_06_04_12_41_32
```

Profile summary:

- Step-time average: **1998.9 ms**
- MXU utilization: **64.1%**
- Total profile time: **111,827.9 ms**
- Idle: **287.9 ms / 0.3%**
- Peak HBM: **26.91 GiB** of 31.25 GiB

Top device buckets:

- Convolution fusion: **60,204.4 ms / 53.8%**
- Custom-call: **29,348.5 ms / 26.2%**
  - `splash_mha_dkv_no_residuals.1`: **15,477.6 ms / 13.8%**
  - `splash_mha_fwd_residuals.2/.3`: **13,870.9 ms / 12.4%**
- Loop fusion: **11,675.8 ms / 10.4%**
- All-reduce-scatter fusion: **3,763.4 ms / 3.4%**

The profile remains effectively identical to v123/v126. The train-step program
hash is the same:

```text
jit_train_step(16296087512277873197)
```

## HLO Dump

Fresh dumps were emitted under:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v127-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-all-experimental-scheduler/hlo/
```

The final train-step module is `module_0267.jit_train_step.cl_854318611`.

- Optimized HLO size: **787,978 bytes**
- SHA256: `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`
- Memory report: **28,862,302,652 bytes / 26.88 GiB**

This matches the v123/v126 optimized train-step HLO exactly. The flagfile
confirms the intended change:

```text
--xla_tpu_enable_all_experimental_scheduler_features=false
--xla_tpu_enable_scheduler_memory_pressure_tracking='true'
--xla_latency_hiding_scheduler_enable_selective_resources='true'
--xla_tpu_enable_latency_hiding_layer_scheduler='false'
--xla_tpu_enable_sparse_core_collective_aggregator='false'
--xla_tpu_all_gather_collective_matmul_mode=NONE
--xla_tpu_reduce_scatter_collective_matmul_mode=NONE
--xla_tpu_scoped_vmem_limit_kib=81920
--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'
```

The trainer-side Pallas dump also shows Tokamax Splash metadata with
`use_experimental_scheduler=false`, so the intended scheduler-bundle ablation was
actually applied even though the final train-step HLO/profile stayed identical.

## Verdict

**Supported tie / simplification.** Disabling the broad all-experimental
scheduler bundle on the simplified v126 batch-4 collective-matmul-none baseline
does not improve the frontier, but it also does not harm the final train-step
program. Carry this as the simplified baseline for the next one-factor scheduler
probe.

The named frontier remains v123 for best observed step (56,599 tok/s), while
v127 is an equivalent simplified-stack confirmation with the highest in-band
worker0 average so far (56,453 tok/s). Next test:
`--xla_tpu_enable_scheduler_memory_pressure_tracking=false` from the v127 stack.
