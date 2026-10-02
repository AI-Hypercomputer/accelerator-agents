---
title: "Qwen3 JAX v208 - SparseCore collective aggregator relaunch"
type: experiment
hypothesis: "Once all FSDP collectives are offloaded to SparseCore, enabling the SparseCore collective aggregator may further reduce collective overhead or scheduling gaps."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-all-collectives-aggregator-rerun
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collective-aggregator, collectives, all-gather, reduce-scatter, all-reduce, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8, rerun]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v208 - SparseCore collective aggregator relaunch

## Hypothesis under test

**Hypothesis**: v205/v206 confirmed the all-gather + reduce-scatter +
all-reduce SparseCore offload stack at 59,351 tok/s / 42.6% MFU. With
offload active, enabling the SparseCore collective aggregator may group or
schedule offloaded collectives more effectively and improve throughput.

**Mechanism**: Clean relaunch of the v207 hypothesis after v207 was killed
inconclusively. Keep the confirmed v206 stack fixed and change only:

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

GKE workload `alekseyv-qwen3-v208-noscan-scagg-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v208-no-scan-sparsecore-aggregator-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA placement/scheduling experiment only
and preserves model math.

## Results

Completed cleanly on `alekseyv-qwen3-v208-noscan-scagg-r` with `EXIT_CODE=0`
on both workers.

- Worker 0 reported **59,372 tok/s** aggregate, **7,422 tok/s/chip**, and
  **42.6% MFU**.
- Worker 1 reported **59,355 tok/s** aggregate, **7,419 tok/s/chip**, and
  **42.6% MFU**.
- Final measured steady steps after profile export stayed in the v206 band:
  worker0 steps 15-19 were 59,538, 59,450, 59,398, 59,365, and 59,327 tok/s;
  worker1 steps 15-19 were 59,539, 59,450, 59,397, 59,360, and 59,325 tok/s.
- Loss decreased normally through step 19, ending at **12.0459**.

This is a very small top-line lift over the v206 confirmed best-worker average
of 59,351 tok/s, but the optimized train-step HLO is byte-identical to v205, so
the observed gain is runtime/scheduling-level rather than a new compiled
program. v210 later reproduced the result at 59,370 tok/s / 42.6% MFU on the
best worker, so the aggregator flag is now rerun-confirmed as a marginal
frontier component.

## Profile

Profile capture completed on both hosts:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v208-no-scan-sparsecore-aggregator-rerun/plugins/profile/2026_06_05_01_59_18/gke-tpu-964065d9-9vh8.{trace.json.gz,xplane.pb}`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v208-no-scan-sparsecore-aggregator-rerun/plugins/profile/2026_06_05_01_59_18/gke-tpu-964065d9-c180.{trace.json.gz,xplane.pb}`

## HLO Dump

Fresh train-step HLO was emitted:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v208-no-scan-sparsecore-aggregator-rerun/hlo/module_0190.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `d67433daa35f81edbfbd0fb25d79bd4cd210fc9697bd39133cd50dfe11da9573`
- Compiled memory: `29700367320` bytes / **27.66 GiB**.

The SHA and memory are identical to the v205 reference program, despite
`--xla_tpu_enable_sparse_core_collective_aggregator=true`.

## Verdict

**Supported and rerun-confirmed by v210.** v208 completed cleanly and reported
a new best-worker average of **59,372 tok/s / 42.6% MFU**; v210 reproduced the
band at **59,370 tok/s / 42.6% MFU**. The lift over v206 is tiny (+0.03-0.04%)
and the optimized train-step HLO is unchanged, so carry this only as a marginal
runtime scheduling win.
