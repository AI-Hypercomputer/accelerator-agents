---
title: "Qwen3 JAX v126 - shard acts s8k bs4 tokamax splash collective matmul none layeragg false"
type: experiment
hypothesis: "The v124 omission test fell back to auto; explicitly disabling the layer-scheduler plus sparse-core aggregator pair may reveal whether the pair is truly required under collective-matmul-none."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, ablation, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v126 - shard acts s8k bs4 tokamax splash collective matmul none layeragg false

## Hypothesis under test

**Hypothesis**: v124 only omitted the explicit layer-scheduler plus sparse-core
aggregator flags, and the compiled flagfile fell back to `auto` while producing
the same HLO as v123. Forcing both flags to `false` will test whether the pair is
actually needed under the collective-matmul-none batch-4 frontier.

**Mechanism**: Start from the confirmed v123 stack and change only:

```text
--xla_tpu_enable_latency_hiding_layer_scheduler=false
--xla_tpu_enable_sparse_core_collective_aggregator=false
```

Keep selective resources and the collective-matmul-none pair enabled:

```text
--xla_latency_hiding_scheduler_enable_selective_resources=true
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: Clean completion. If the HLO/profile remain identical to
v123/v124, the pair is truly irrelevant under collective-matmul-none. If the HLO
changes, throughput determines whether the pair is still beneficial.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below the confirmed v118/v123/v124 batch-4 band.

## Setup

GKE workload `alekseyv-qwen3-v126-bs4-layerfalse`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v126-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-layeragg-false`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v123 stack with layer-scheduler and sparse-core aggregator forced false.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers (`EXIT_CODE=0`, no pod restarts).

| Worker | Avg throughput | Tok/s/chip | Approx MFU | Best observed step | Final loss |
|--------|----------------|------------|------------|--------------------|------------|
| 0 | 56,448 tok/s | 7,056 | 40.5% | 56,574 tok/s | 12.0461 |
| 1 | 56,428 tok/s | 7,054 | 40.5% | 56,579 tok/s | 12.0461 |

This reproduces the v118/v123/v124 throughput band. Explicitly disabling the
layer-scheduler plus sparse-core aggregator pair is valid and does not change
the final train-step HLO.

## Profile

XProf run:

```text
2026-06-04-qwen3-jax-v126-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-layeragg-false/2026_06_04_12_30_04
```

Summary:

- XProf step-time average: 1999.4 ms.
- Total device time: 111,845.3 ms.
- Idle: 296.5 ms / 0.3%.
- Train-step program: `jit_train_step(16296087512277873197)`, same as v118/v123/v124.
- MXU utilization: 63.9%.
- Peak HBM: 26.91 GiB (`21.118` GiB stack, `5.7894` GiB heap, `4.3388` GiB free).

Top train-step buckets:

| Bucket | Time | Share |
|--------|------|-------|
| convolution fusion | 60,225.3 ms | 53.8% |
| custom-call | 29,348.4 ms | 26.2% |
| splash DKV | 15,477.6 ms | 13.8% |
| splash forward residuals | 13,870.8 ms | 12.4% |
| loop fusion | 11,672.6 ms | 10.4% |
| all-reduce-scatter fusion | 3,761.7 ms | 3.4% |

## HLO Dump

The optimized train-step HLO is byte-identical to v118/v123/v124:

- Optimized train-step module: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Optimized train-step text size: 787,978 bytes.
- HLO content hash: `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`.
- HLO prefix object count: 1,611.
- Memory report total bytes: 28,862,302,652 bytes / 26.88 GiB.
- Flagfile confirms both flags were explicitly disabled:

```text
--xla_tpu_enable_latency_hiding_layer_scheduler='false'
--xla_tpu_enable_sparse_core_collective_aggregator='false'
--xla_latency_hiding_scheduler_enable_selective_resources='true'
--xla_tpu_all_gather_collective_matmul_mode=NONE
--xla_tpu_reduce_scatter_collective_matmul_mode=NONE
```

## Verdict

**Supported tie.** The layer-scheduler plus sparse-core aggregator pair is not
needed under the batch-4 collective-matmul-none frontier. The simplified stack
with both flags explicitly `false` preserves the same train-step HLO, memory
profile, and throughput band. This is not a new performance frontier, but it
closes the v124 ambiguity and allows future probes to use the simpler scheduler
baseline.
