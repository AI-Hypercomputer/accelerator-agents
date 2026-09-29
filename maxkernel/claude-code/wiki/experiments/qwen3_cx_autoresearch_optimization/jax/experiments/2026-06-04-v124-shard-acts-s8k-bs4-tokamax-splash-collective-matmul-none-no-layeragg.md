---
title: "Qwen3 JAX v124 - shard acts s8k bs4 tokamax splash collective matmul none no layeragg"
type: experiment
hypothesis: "After collective-matmul modes are disabled, the earlier layer-scheduler plus sparse-core aggregator pair may be redundant or harmful."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, ablation, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v124 - shard acts s8k bs4 tokamax splash collective matmul none no layeragg

## Hypothesis under test

**Hypothesis**: The layer-scheduler plus sparse-core collective aggregator pair
helped the pre-v116 51k tok/s scheduler frontier, but may be redundant after the
collective-matmul lowering modes were disabled. Removing the pair from the
confirmed v123 batch-4 stack may reduce scheduling overhead or keep the same
program shape without losing throughput.

**Mechanism**: Start from the confirmed v123 stack and remove only:

```text
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_enable_sparse_core_collective_aggregator=true
```

Keep the v123 batch/global shape and the collective-matmul-none pair:

```text
--batch_size=4
--xla_tpu_scoped_vmem_limit_kib=81920
--xla_tpu_scheduler_percent_shared_memory_limit=100
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: Clean completion with throughput at or above v123's
56.44k tok/s / 40.5% MFU band, or a same-band result with a simpler scheduler
stack.

**Falsification criterion**: Compile/runtime failure, invalid loss, or a clean
throughput regression below the confirmed v118/v123 batch-4 band.

## Setup

GKE workload `alekseyv-qwen3-v124-bs4-nolayeragg`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v124-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-layeragg`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v123 stack minus the layer-scheduler and sparse-core aggregator pair.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers (`EXIT_CODE=0`, no pod restarts).

| Worker | Avg throughput | Tok/s/chip | Approx MFU | Best observed step | Final loss |
|--------|----------------|------------|------------|--------------------|------------|
| 0 | 56,447 tok/s | 7,056 | 40.5% | 56,569 tok/s | 12.0461 |
| 1 | 56,429 tok/s | 7,054 | 40.5% | 56,576 tok/s | 12.0461 |

This is effectively the same throughput band as v118/v123. Removing the explicit
layer-scheduler plus sparse-core aggregator flags did not create a measurable
runtime improvement.

## Profile

XProf run:

```text
2026-06-04-qwen3-jax-v124-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-layeragg/2026_06_04_12_05_51
```

Summary:

- XProf step-time average: 1999.4 ms.
- Total device time: 111,854.0 ms.
- Idle: 295.5 ms / 0.3%.
- Train-step program: `jit_train_step(16296087512277873197)`, same as v118/v123.
- MXU utilization: 64.1%.
- Peak HBM: 26.91 GiB (`21.118` GiB stack, `5.7894` GiB heap, `4.3388` GiB free).

Top train-step buckets:

| Bucket | Time | Share |
|--------|------|-------|
| convolution fusion | 60,205.3 ms | 53.8% |
| custom-call | 29,348.8 ms | 26.2% |
| splash DKV | 15,477.6 ms | 13.8% |
| splash forward residuals | 13,871.2 ms | 12.4% |
| loop fusion | 11,666.6 ms | 10.4% |
| all-reduce-scatter fusion | 3,762.5 ms | 3.4% |

## HLO Dump

The run emitted fresh HLO dumps and the optimized train-step text is
byte-identical to v118/v123:

- Optimized train-step modules: `module_0264.jit_train_step...` and
  `module_0267.jit_train_step...`.
- Optimized train-step text size: 787,978 bytes.
- HLO content hash: `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`.
- HLO prefix object count: 2,226.
- Memory report total bytes: 28,862,302,652 bytes / 26.88 GiB.
- Flagfile confirms both collective-matmul modes are `NONE`, scoped VMEM is
  81920 KiB, scheduler shared-memory limit is 100, and selective resources
  remain enabled. The omitted pair appears as default `auto`:

```text
--xla_tpu_enable_latency_hiding_layer_scheduler=auto
--xla_tpu_enable_sparse_core_collective_aggregator=auto
```

## Verdict

**Supported tie, not adopted as a new frontier.** v124 confirms the explicit
layer-scheduler plus sparse-core aggregator flags are not required to reproduce
the current batch-4 collective-matmul-none HLO/profile, but it does not beat
v123. Keep v123 as the named frontier and continue scheduler ablations from the
same 56.44k tok/s / 40.5% MFU band.
