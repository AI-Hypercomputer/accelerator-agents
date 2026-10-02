---
title: "Qwen3 JAX v219 - no-scan SparseCore AG/AR-only offload"
type: experiment
hypothesis: "The v210 frontier bundled SparseCore offload for all-gather, reduce-scatter, and all-reduce; reduce-scatter offload may be neutral once all-gather/all-reduce offload and the aggregator are enabled."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-ag-ar-only
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, all-gather, reduce-scatter, all-reduce, ablation, collective-aggregator, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v219 - no-scan SparseCore AG/AR-only offload

## Hypothesis under test

**Hypothesis**: [v218](2026-06-05-v218-no-scan-sparsecore-ag-rs-only.md)
showed that removing only all-reduce SparseCore offload reduces the visible
async-done bucket but still regresses throughput, so all-reduce offload should
stay enabled. The remaining isolate is reduce-scatter: it may be less important
than all-gather/all-reduce under the v210 SparseCore aggregator stack.

**Mechanism**: Keep the confirmed [v210](2026-06-05-v210-no-scan-sparsecore-aggregator-rerun3.md)
frontier fixed, including no-scan, remat, no MLP-intermediate `_sac`, Tokamax
Splash, MaxText CE, activation sharding, collective-matmul modes `none`, scoped
VMEM 100352 KiB, all-gather/all-reduce SparseCore offload, and SparseCore
collective aggregator. Change only the reduce-scatter offload bundle:

```text
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false
--xla_tpu_enable_reduce_scatter_offload_tracing=false
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true
```

**Predicted signal**: Support requires clean compile/loss and best-worker
throughput above the confirmed v210 band of **59,370-59,372 tok/s / 42.6% MFU**,
or a profile-level reduction in collective/async tail time that does not move
cost into convolution/custom-call buckets.

**Falsification criterion**: Compile/runtime failure, invalid loss, throughput
at or below the v210 band, increased async tail, or an HLO/profile change that
adds memory/collective cost without a throughput gain.

## Setup

GKE workload `alekseyv-qwen3-v219-sc-agar`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v219-no-scan-sparsecore-ag-ar-only`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA placement/scheduling experiment only
and preserves model math.

## Result

Launched as `alekseyv-qwen3-v219-sc-agar` on
`<your-cluster>` at 2026-06-05 04:47 UTC and completed cleanly
with `EXIT_CODE=0` on both workers.

Pods:

- `alekseyv-qwen3-v219-sc-agar-slice-job-0-0-46t45` on `gke-tpu-46dd3e54-rkr2`
- `alekseyv-qwen3-v219-sc-agar-slice-job-0-1-twht4` on `gke-tpu-46dd3e54-g30s`

- Worker 0 reported **58,238 tok/s** aggregate, **7,280 tok/s/chip**, and
  **41.8% MFU**.
- Worker 1 reported **58,269 tok/s** aggregate, **7,284 tok/s/chip**, and
  **41.8% MFU**.
- Best visible steady step was only **58,406 tok/s**, well below the v210 band
  of 59,370-59,372 tok/s.
- Loss decreased normally through step 19, ending at **12.0463**.

This is a hard throughput regression, so reduce-scatter SparseCore offload is
part of the carried v210 bundle.

## Profile

Profile capture completed on both hosts:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v219-no-scan-sparsecore-ag-ar-only/plugins/profile/2026_06_05_04_52_42/gke-tpu-46dd3e54-rkr2.{trace.json.gz,xplane.pb}`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v219-no-scan-sparsecore-ag-ar-only/plugins/profile/2026_06_05_04_52_42/gke-tpu-46dd3e54-g30s.{trace.json.gz,xplane.pb}`

XProf run:

- `2026-06-05-qwen3-jax-v219-no-scan-sparsecore-ag-ar-only/2026_06_05_04_52_42`

Profile summary:

- Step time average: **4511.3 ms**
- MXU utilization: **65.8%**
- Peak memory: **30.99 GiB** on a 31.25 GiB device

The op profile shows `jit_train_step` taking **108,018.8 ms** across the
captured window versus v210's **106,155.5 ms**. Dominant compute fusions do not
grow; `convolution fusion` actually falls to **56,860.5 ms / 52.5%** from
v210's 57,712.8 ms / 54.2%. The regression comes from a worse collective/fusion
schedule: the profile exposes new all-reduce-containing fusion buckets such as
`fusion.39 and its duplicate(s)` at **1,145.8 ms / 1.1%** and `fusion.3` at
**310.6 ms / 0.3%**, while overall MXU drops.

## HLO Dump

Fresh train-step HLO was emitted:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v219-no-scan-sparsecore-ag-ar-only/hlo/xla-dump/module_0294.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `5c28221460c5446a0888b89db036f6bc11f47e9cacce9c8416766009bb9555ca`
- Size: **90,801 lines / 15,650,659 bytes**
- Compiled memory: `29693976024` bytes / **27.65 GiB**
- HLO prefix size: **127,521,408 bytes**

Quick text counts:

| Pattern | Count |
| --- | ---: |
| `all-gather` | 5,394 |
| `all-reduce` | 965 |
| `reduce-scatter` | 372 |
| `collective-permute` | 3 |
| `convolution` | 2,255 |
| `fusion` | 9,885 |
| `custom-call` | 2,283 |
| `copy` | 10,453 |
| `async-start` | 724 |
| `async-done` | 724 |

## Verdict

**Refuted.** Do not remove SparseCore reduce-scatter offload. v219 preserves
model math and completes cleanly, but it drops about **1.1k tok/s** versus the
confirmed v210 frontier and lowers MXU. Keep reduce-scatter SparseCore offload
enabled in the frontier.

Together with [v218](2026-06-05-v218-no-scan-sparsecore-ag-rs-only.md), this
shows both all-reduce and reduce-scatter offload are needed. The remaining
cheap isolate is all-gather: keep reduce-scatter and all-reduce on SparseCore,
move only all-gather back to async collective fusion, and expect a likely
regression if all-gather is the largest contributor to the v205/v206 lift.
