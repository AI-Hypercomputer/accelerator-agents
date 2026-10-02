---
title: "Qwen3 JAX v220 - no-scan SparseCore RS/AR-only offload"
type: experiment
hypothesis: "The v210 frontier bundled SparseCore offload for all-gather, reduce-scatter, and all-reduce; all-gather offload may be the remaining required member after v218/v219 showed all-reduce and reduce-scatter are needed."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-only
status: provisional-supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, all-gather, reduce-scatter, all-reduce, ablation, collective-aggregator, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v220 - no-scan SparseCore RS/AR-only offload

## Hypothesis under test

**Hypothesis**: [v218](2026-06-05-v218-no-scan-sparsecore-ag-rs-only.md)
and [v219](2026-06-05-v219-no-scan-sparsecore-ag-ar-only.md) showed that
all-reduce and reduce-scatter SparseCore offload are both part of the carried
v210 frontier. The remaining isolate is all-gather: if all-gather offload is
also required, returning only all-gather to async collective fusion should
regress versus v210.

**Mechanism**: Keep the confirmed [v210](2026-06-05-v210-no-scan-sparsecore-aggregator-rerun3.md)
frontier fixed, including no-scan, remat, no MLP-intermediate `_sac`, Tokamax
Splash, MaxText CE, activation sharding, collective-matmul modes `none`, scoped
VMEM 100352 KiB, reduce-scatter/all-reduce SparseCore offload, and SparseCore
collective aggregator. Change only the all-gather offload bundle:

```text
--xla_tpu_enable_sparse_core_collective_offload_all_gather=false
--xla_tpu_enable_all_gather_offload_tracing=false
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
```

**Predicted signal**: Support requires clean compile/loss and best-worker
throughput above the confirmed v210 band of **59,370-59,372 tok/s / 42.6% MFU**,
or a profile-level reduction in collective/async tail time that does not move
cost into convolution/custom-call buckets.

**Falsification criterion**: Compile/runtime failure, invalid loss, throughput
at or below the v210 band, increased async tail, or an HLO/profile change that
adds memory/collective cost without a throughput gain.

## Setup

GKE workload `alekseyv-qwen3-v220-sc-rsar`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v220-no-scan-sparsecore-rs-ar-only`
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

Launched as `alekseyv-qwen3-v220-sc-rsar` on
`<your-cluster>` at 2026-06-05 05:01 UTC and completed cleanly
with `EXIT_CODE=0` on both workers.

Pods:

- `alekseyv-qwen3-v220-sc-rsar-slice-job-0-0-tzvlv` on `gke-tpu-964065d9-c180`
- `alekseyv-qwen3-v220-sc-rsar-slice-job-0-1-fvbdq` on `gke-tpu-964065d9-9vh8`

Throughput:

| Worker | Avg steps 2-19 | Best step 2-19 | Avg steps 15-19 | Tok/s/chip from avg 2-19 | Nominal MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | **59,441 tok/s** | **59,645 tok/s** | **59,481 tok/s** | **7,430** | **42.7%** |
| 1 | **59,424 tok/s** | **59,644 tok/s** | **59,480 tok/s** | **7,428** | **42.7%** |

Loss decreased normally through step 19, ending at **12.0459**. The result is
above the v208/v210 confirmed 59,370-59,372 tok/s band, but the margin is small
enough that it needs an exact rerun before treating the all-gather SparseCore
offload removal as carried.

## Profile

Profile capture completed on both hosts:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v220-no-scan-sparsecore-rs-ar-only/plugins/profile/2026_06_05_05_06_54/gke-tpu-964065d9-c180.{trace.json.gz,xplane.pb}`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v220-no-scan-sparsecore-rs-ar-only/plugins/profile/2026_06_05_05_06_54/gke-tpu-964065d9-9vh8.{trace.json.gz,xplane.pb}`

XProf run:

- `2026-06-05-qwen3-jax-v220-no-scan-sparsecore-rs-ar-only/2026_06_05_05_06_54`

Profile summary:

- Step time average: **4428.4 ms** versus v210's indexed **4433.3 ms**
- MXU utilization: **67.5%** versus v210's **67.4%**
- Peak memory: **30.93 GiB** versus v210's **31.06 GiB**
- `jit_train_step` program time: **106,044.4 ms** versus v210's **106,155.5 ms**
- Dominant op buckets:
  - `convolution fusion`: **57,373.5 ms / 54.0%** versus v210's 57,712.8 ms / 54.2%
  - `custom-call`: **29,419.8 ms / 27.7%**, essentially tied with v210
  - `loop fusion`: **10,745.6 ms / 10.1%**, slightly lower than v210
  - `async-done`: **1,087.6 ms / 1.0%**, much lower than v210's 3,646.9 ms / 3.4%
  - exposed direct `all-gather`: **218.7 ms / 0.2%**

The profile supports the log-level gain: returning all-gather to async
collective fusion makes the HLO larger but cuts the visible async-done tail
substantially and slightly improves step time/MXU.

## HLO Dump

Fresh train-step HLO was emitted:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v220-no-scan-sparsecore-rs-ar-only/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `a33c7940efea974ff9a7dcf2704fa4d08f79e5d82e1821a59a553805ceca81d5`
- Size: **119,545 lines / 19,954,794 bytes**
- Compiled memory: `29704457684` bytes / **27.66 GiB**
- HLO prefix size: **896 objects / 157.05 MiB**

Flagfile confirms the intended isolate:

```text
--xla_tpu_enable_sparse_core_collective_offload_all_gather='false'
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter='true'
--xla_tpu_enable_sparse_core_collective_offload_all_reduce='true'
--xla_tpu_enable_async_collective_fusion_fuse_all_gather='true'
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false
--xla_tpu_enable_sparse_core_collective_aggregator='true'
```

Quick text counts:

| Pattern | Count |
| --- | ---: |
| `all-gather` | 7,039 |
| `all-reduce` | 165 |
| `reduce-scatter` | 2,774 |
| `collective-permute` | 6 |
| `convolution` | 3,071 |
| `fusion` | 31,065 |
| `custom-call` | 3,837 |
| `copy` | 22,122 |
| `async-start` | 382 |
| `async-done` | 382 |

## Verdict

**Provisional supported.** v220 disproves the expectation that all-gather
SparseCore offload is required. Keeping reduce-scatter/all-reduce on SparseCore
while returning all-gather to async collective fusion gives the best observed
native-JAX result so far: **59,441 tok/s / 7,430 tok/s/chip / 42.7% MFU** on
worker0, with XProf step time also slightly better than v210.

Because the improvement over the confirmed v208/v210 band is only about
0.1-0.2%, run an exact rerun before carrying this as the confirmed frontier.
