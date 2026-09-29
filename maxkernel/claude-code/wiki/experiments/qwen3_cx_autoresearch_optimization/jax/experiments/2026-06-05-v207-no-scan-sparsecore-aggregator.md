---
title: "Qwen3 JAX v207 - SparseCore collective aggregator on confirmed SC-offload frontier"
type: experiment
hypothesis: "Once all FSDP collectives are offloaded to SparseCore, enabling the SparseCore collective aggregator may further reduce collective overhead or scheduling gaps."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-all-collectives-aggregator
status: inconclusive
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collective-aggregator, collectives, all-gather, reduce-scatter, all-reduce, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v207 - SparseCore collective aggregator on confirmed SC-offload frontier

## Hypothesis under test

**Hypothesis**: v205/v206 confirmed the all-gather + reduce-scatter +
all-reduce SparseCore offload stack at 59,351 tok/s / 42.6% MFU. The
SparseCore collective aggregator was previously a no-op in v158 when
SparseCore collective offload was disabled. With offload now active, enabling
the aggregator may group or schedule offloaded collectives more effectively and
improve throughput.

**Mechanism**: Keep the confirmed v206 stack fixed and change only:

```text
--xla_tpu_enable_sparse_core_collective_aggregator=true
```

from the v205/v206 value `false`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the confirmed v206 59,351 tok/s / 42.6% MFU frontier.

**Falsification criterion**: Compile/runtime failure, invalid loss, no HLO
change with no throughput gain, or clean completion at or below the confirmed
frontier.

## Setup

GKE workload `alekseyv-qwen3-v207-noscan-scagg`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v207-no-scan-sparsecore-aggregator`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA placement/scheduling experiment only
and preserves model math.

## Results

Inconclusive. The workload started and emitted early helper-module HLO, but no
CE/step output was captured. A direct `kubectl get jobset` check returned
NotFound while XPK still showed the workload quota-reserved, so I treated it as
an orphaned/inconsistent control-plane state and deleted it with
`xpk workload delete`.

The delete then found and removed the JobSet, so this was an operator cleanup
of an ambiguous running state rather than a model/runtime verdict. Do not use
v207 as evidence for or against the aggregator flag.

## Profile

None captured.

## HLO Dump

Only early helper-module HLO appeared under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v207-no-scan-sparsecore-aggregator/hlo/`

No train-step HLO was reached.

## Verdict

**Inconclusive.** Relaunch the same hypothesis with a fresh workload/run
directory.
