---
title: "Qwen3 JAX v125 - shard acts s8k bs4 tokamax splash layeragg collective matmul none no selective resources"
type: experiment
hypothesis: "After collective-matmul modes are disabled, selective scheduler resources may no longer help the final batch-4 train-step schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, selective-resources, ablation, layer-scheduler, sparse-core-collective-aggregator, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v125 - shard acts s8k bs4 tokamax splash layeragg collective matmul none no selective resources

## Hypothesis under test

**Hypothesis**: Selective scheduler resources helped the pre-v116 scheduler
frontier, but may be redundant or harmful after the collective-matmul lowering
modes were disabled and the batch-4 train-step HLO stabilized. Disabling
selective resources may produce the same HLO/profile or slightly improve runtime
scheduling.

**Mechanism**: Start from the confirmed v123 stack and change only:

```text
--xla_latency_hiding_scheduler_enable_selective_resources=false
```

Keep the v123 batch/global shape, layer-scheduler pair, and collective-matmul
none pair:

```text
--batch_size=4
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_enable_sparse_core_collective_aggregator=true
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: Clean completion with throughput at or above v123's
56.44k tok/s / 40.5% MFU band, or a same-band result with identical HLO that
proves selective resources are unnecessary.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below the confirmed v118/v123/v124 batch-4 band.

## Setup

GKE workload `alekseyv-qwen3-v125-bs4-noselres`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v125-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none-no-selective-resources`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v123 stack with selective resources set to false.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers (`EXIT_CODE=0`, no pod restarts).

| Worker | Avg throughput | Tok/s/chip | Approx MFU | Best observed step | Final loss |
|--------|----------------|------------|------------|--------------------|------------|
| 0 | 56,387 tok/s | 7,048 | 40.4% | 56,524 tok/s | 12.0458 |
| 1 | 56,369 tok/s | 7,046 | 40.4% | 56,526 tok/s | 12.0458 |

This is a small but repeatable regression from v123/v124's 56.43-56.45k tok/s
band. Selective resources should stay enabled.

## Profile

XProf run:

```text
2026-06-04-qwen3-jax-v125-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none-no-selective-resources/2026_06_04_12_18_17
```

Summary:

- XProf step-time average: 2001.0 ms.
- Total device time: 111,948.0 ms.
- Idle: 272.4 ms / 0.2%.
- Train-step program: `jit_train_step(12520399088891659285)`, changed from the
  v118/v123/v124 program hash.
- MXU utilization: 64.0%.
- Peak HBM: 26.91 GiB (`21.1181` GiB stack, `5.7899` GiB heap, `4.3381` GiB free).

Top train-step buckets:

| Bucket | Time | Share |
|--------|------|-------|
| convolution fusion | 60,086.4 ms | 53.7% |
| custom-call | 29,348.1 ms | 26.2% |
| splash DKV | 15,477.7 ms | 13.8% |
| splash forward residuals | 13,870.3 ms | 12.4% |
| loop fusion | 11,714.6 ms | 10.5% |
| all-reduce-scatter fusion | 3,762.4 ms | 3.4% |

The regression is small, but the changed program spends slightly more absolute
time in loop fusion while not improving custom-call, collectives, or memory.

## HLO Dump

The run emitted a real changed train-step HLO:

- Optimized train-step module: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Optimized train-step text size: 788,403 bytes.
- HLO content hash: `c87d6be51ac696104580c7500f3970e2073a4533e20a1b34649b3cc9f2f87ba7`.
- HLO prefix object count: 1,619.
- Memory report total bytes: 28,862,417,340 bytes / 26.88 GiB.
- Flagfile confirms the one-factor change:

```text
--xla_latency_hiding_scheduler_enable_selective_resources='false'
--xla_tpu_enable_latency_hiding_layer_scheduler='true'
--xla_tpu_enable_sparse_core_collective_aggregator='true'
--xla_tpu_all_gather_collective_matmul_mode=NONE
--xla_tpu_reduce_scatter_collective_matmul_mode=NONE
```

## Verdict

**Refuted.** Disabling selective resources selects a real different train-step
HLO and regresses throughput to 56.37-56.39k tok/s without reducing HBM or
custom-call time. Keep selective resources enabled in the current frontier.
