---
title: "Qwen3 JAX v218 - no-scan SparseCore AG/RS-only offload"
type: experiment
hypothesis: "The v210 frontier bundled SparseCore offload for all-gather, reduce-scatter, and all-reduce; all-reduce offload may be neutral or slightly harmful once the aggregator is enabled."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-ag-rs-only
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, all-gather, reduce-scatter, all-reduce, ablation, collective-aggregator, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v218 - no-scan SparseCore AG/RS-only offload

## Hypothesis under test

**Hypothesis**: [v205](2026-06-05-v205-no-scan-sparsecore-all-collectives.md)
and [v206](2026-06-05-v206-no-scan-sparsecore-all-collectives-rerun.md)
confirmed a large no-scan gain by moving all-gather, reduce-scatter, and
all-reduce collectives onto SparseCore. [v208](2026-06-05-v208-no-scan-sparsecore-aggregator-rerun.md)
and [v210](2026-06-05-v210-no-scan-sparsecore-aggregator-rerun3.md) then
confirmed a marginal aggregator lift. The all-reduce part of the SparseCore
bundle has not been isolated; it may be unnecessary overhead if AG/RS offload
captures most of the benefit.

**Mechanism**: Keep the confirmed v210 frontier fixed, including no-scan,
remat, no MLP-intermediate `_sac`, Tokamax Splash, MaxText CE, activation
sharding, collective-matmul modes `none`, scoped VMEM 100352 KiB, AG/RS
SparseCore offload, and SparseCore collective aggregator. Change only the
all-reduce offload bundle:

```text
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false
--xla_tpu_enable_all_reduce_offload_tracing=false
--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true
```

**Predicted signal**: Support requires clean compile/loss and best-worker
throughput above the confirmed v210 band of **59,370-59,372 tok/s / 42.6% MFU**,
or a profile-level reduction in async-done/data-formatting time without moving
cost into convolution/custom-call buckets.

**Falsification criterion**: Compile/runtime failure, invalid loss, throughput
at or below the v210 band, increased async tail, or an HLO/profile change that
adds memory/collective cost without a throughput gain.

## Setup

GKE workload `alekseyv-qwen3-v218-sc-agrs`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v218-no-scan-sparsecore-ag-rs-only`
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

Completed cleanly on `alekseyv-qwen3-v218-sc-agrs` with `EXIT_CODE=0` on both
workers, normal loss, profile capture, and uploaded HLO.

- Worker 0 reported **59,333 tok/s** aggregate, **7,417 tok/s/chip**, and
  **42.5% MFU**.
- Worker 1 reported **59,359 tok/s** aggregate, **7,420 tok/s/chip**, and
  **42.6% MFU**.
- Best visible steady step was **59,543 tok/s**, but the measured averages are
  below the confirmed v210 band of 59,370-59,372 tok/s.
- Loss decreased normally through step 19, ending at **12.0461**.

The runtime ablation is valid but does not improve the frontier. Moving
all-reduce back out of SparseCore offload slightly reduces the visible
`async-done` bucket in XProf, but the saving is too small to overcome the
runtime throughput regression.

## Profile

Profile capture completed on both hosts:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v218-no-scan-sparsecore-ag-rs-only/plugins/profile/2026_06_05_04_39_11/gke-tpu-964065d9-9vh8.{trace.json.gz,xplane.pb}`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v218-no-scan-sparsecore-ag-rs-only/plugins/profile/2026_06_05_04_39_11/gke-tpu-964065d9-c180.{trace.json.gz,xplane.pb}`

XProf run:

- `2026-06-05-qwen3-jax-v218-no-scan-sparsecore-ag-rs-only/2026_06_05_04_39_11`

Compared with v210, the `jit_train_step` op profile is almost identical in the
dominant buckets:

| Bucket | v218 | v210 |
| --- | ---: | ---: |
| `convolution fusion` | 57,710 ms / 54.2% | 57,713 ms / 54.2% |
| `async-done` | 3,393 ms / 3.2% | 3,646 ms / 3.4% |

The removed all-reduce offload does eliminate the v210 top async-done entry
`all-reduce.660...call-done` (~298 ms), but total train-step program time stays
flat in the profile and the trainer-reported throughput regresses. Peak XProf
memory was **29.90 GiB** versus v210's **31.06 GiB**, so the ablation frees
runtime headroom but not useful wall time.

## HLO Dump

Fresh train-step HLO was emitted:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v218-no-scan-sparsecore-ag-rs-only/hlo/xla-dump/module_0294.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `eeb040b12d05e32196f22516547a22d84a9343d6681871d4ee7962c1a63308df`
- Size: **94,154 lines / 16,537,770 bytes**
- Compiled memory: `29700334552` bytes / **27.66 GiB**

The comparable optimized HLO is not byte-identical to v210
(`d67433daa...`), but compiled memory remains effectively unchanged. Quick text
counts:

| Pattern | Count |
| --- | ---: |
| `all-gather` | 5,393 |
| `all-reduce` | 153 |
| `reduce-scatter` | 1,766 |
| `collective-permute` | 3 |
| `convolution` | 2,276 |
| `fusion` | 9,652 |
| `custom-call` | 3,039 |
| `copy` | 10,956 |
| `async-start` | 977 |
| `async-done` | 977 |

## Verdict

**Refuted.** Do not remove SparseCore all-reduce offload from the frontier. The
all-reduce async tail is visible in v210, but offloading it is still at least
neutral-to-beneficial for end-to-end throughput. Keep v210's full
all-gather/reduce-scatter/all-reduce SparseCore offload bundle plus aggregator.

The next low-cost ablation is the complementary reduce-scatter isolate: keep
all-gather and all-reduce on SparseCore, move only reduce-scatter back to async
fusion, and test whether the v210 benefit depends on reduce-scatter offload.
