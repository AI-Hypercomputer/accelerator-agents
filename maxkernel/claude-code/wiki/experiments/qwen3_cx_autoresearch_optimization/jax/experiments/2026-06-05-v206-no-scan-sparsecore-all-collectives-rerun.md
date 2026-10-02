---
title: "Qwen3 JAX v206 - no-scan SparseCore all-collective rerun"
type: experiment
hypothesis: "The v205 all-collective SparseCore-offload lift is stable and not a one-run noise artifact."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-all-collectives-rerun
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, all-gather, reduce-scatter, all-reduce, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8, rerun]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v206 - no-scan SparseCore all-collective rerun

## Hypothesis under test

**Hypothesis**: v205 improved the no-scan frontier from 58,288-58,330 tok/s /
41.8% MFU to 59,338 tok/s / 42.5% MFU by enabling SparseCore offload for
all-gather, reduce-scatter, and all-reduce. An exact rerun should reproduce the
same HLO and remain above the prior v182/v183 frontier.

**Mechanism**: No new mechanism. Exact v205 rerun with a fresh workload name and
fresh GCS run directory.

**Predicted signal**: Support requires clean compile/loss and throughput in the
v205 band, approximately 59.2k-59.4k tok/s / 42.5% MFU, with byte-identical or
clearly related train-step HLO.

**Falsification criterion**: Clean completion that falls back to the v182/v183
band, materially different HLO without a clear reason, invalid loss, or
compile/runtime failure.

## Setup

GKE workload `alekseyv-qwen3-v206-noscan-scall-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v206-no-scan-sparsecore-all-collectives-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

Changed XLA flags are identical to v205:

```text
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_enable_all_gather_offload_tracing=true
--xla_tpu_enable_reduce_scatter_offload_tracing=true
--xla_tpu_enable_all_reduce_offload_tracing=true
--xla_tpu_use_tc_device_shape_on_sc=true
--xla_sc_enable_instruction_fusion=false
--xla_sc_disjoint_spmem=false
--xla_sc_disable_megacore_partitioning=true
```

## Equivalence

No model-code edit is made. This is an exact runtime-flag rerun and preserves
model math.

## Results

Completed cleanly on `alekseyv-qwen3-v206-noscan-scall-r` with `EXIT_CODE=0`
on both workers.

- Worker 0 reported **59,351 tok/s** aggregate, **7,419 tok/s/chip**, and
  **42.6% MFU**.
- Worker 1 reported **59,329 tok/s** aggregate, **7,416 tok/s/chip**, and
  **42.5% MFU**.
- Final measured steady steps after profile export stayed in the v205 band:
  worker0 steps 15-19 were 59,494, 59,411, 59,365, 59,319, and 59,283 tok/s;
  worker1 steps 15-19 were 59,499, 59,417, 59,365, 59,320, and 59,290 tok/s.
- Loss decreased normally through step 19, ending at **12.0459**.

This exactly confirms v205's once-supported lift over the v182/v183 no-scan
frontier. Carry the SparseCore offload for all-gather, reduce-scatter, and
all-reduce as the confirmed JAX frontier.

## Profile

Profile capture completed on both hosts:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v206-no-scan-sparsecore-all-collectives-rerun/plugins/profile/2026_06_05_01_42_37/gke-tpu-964065d9-9vh8.{trace.json.gz,xplane.pb}`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v206-no-scan-sparsecore-all-collectives-rerun/plugins/profile/2026_06_05_01_42_38/gke-tpu-964065d9-c180.{trace.json.gz,xplane.pb}`

## HLO Dump

No fresh HLO dump appeared under the v206 run directory. This is consistent
with a persistent compile-cache hit from v205: step 0 fell from v205's
145.2 s to 22.3 s, and steady-state metrics matched v205 closely.

Use the v205 train-step HLO as the confirmed program reference:

`d67433daa35f81edbfbd0fb25d79bd4cd210fc9697bd39133cd50dfe11da9573`

Compiled memory from v205 was `29700367320` bytes / **27.66 GiB**.

## Verdict

**Supported.** v206 confirms v205. The new confirmed JAX frontier is
**59,351 tok/s / 42.6% MFU** best-worker reported, with both workers in the
59.3k tok/s band. Next hypothesis can safely layer one additional compiler flag
on top of the all-collective SparseCore-offload stack.
