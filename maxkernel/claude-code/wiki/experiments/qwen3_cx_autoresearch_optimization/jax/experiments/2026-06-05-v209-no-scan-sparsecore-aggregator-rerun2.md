---
title: "Qwen3 JAX v209 - SparseCore collective aggregator confirmation rerun"
type: experiment
hypothesis: "The marginal v208 SparseCore collective-aggregator lift over v206 is reproducible and not a one-run noise artifact."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-all-collectives-aggregator-rerun2
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collective-aggregator, collectives, all-gather, reduce-scatter, all-reduce, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8, rerun]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v209 - SparseCore collective aggregator confirmation rerun

## Hypothesis under test

**Hypothesis**: v208 changed only
`--xla_tpu_enable_sparse_core_collective_aggregator=true` on top of the v206
all-collective SparseCore-offload stack and reported a marginal new best of
59,372 tok/s / 42.6% MFU. An exact rerun should reproduce the same throughput
band if the aggregator flag is a real runtime scheduling win.

**Mechanism**: Exact v208 rerun with a fresh workload name and fresh GCS run
directory.

**Predicted signal**: Support requires clean compile/loss and best-worker
throughput at or above the v208 band, roughly 59,36x-59,38x tok/s, with the
same train-step HLO or a clearly related cached program.

**Falsification criterion**: Clean completion back in the v206 band
(approximately 59,33x-59,35x tok/s), invalid loss, missing profile, or
compile/runtime failure.

## Setup

GKE workload `alekseyv-qwen3-v209-noscan-scagg-r2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v209-no-scan-sparsecore-aggregator-rerun2`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an exact runtime-flag rerun and preserves
model math.

## Results

Failed immediately before trainer startup on both workers with `EXIT_CODE=1`.
The rejected flags were:

```text
ERROR: Unknown command line flag 'xla_tpu_enable_layer_wise_all_gather'
ERROR: Unknown command line flag 'xla_tpu_enable_custom_call_combiner'
```

No model code ran, no compile occurred, and no performance signal was produced.
The intended flags should have used the accepted v208 names
`--xla_tpu_enable_latency_hiding_layer_scheduler=false` and
`--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`.

## Profile

None. The workload failed before profiling.

## HLO Dump

None. The workload failed before compilation.

## Verdict

**Invalid.** Launch-command error only. Use v210 as the corrected confirmation
rerun.
