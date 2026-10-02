---
title: "Qwen3 JAX v211 - concurrent SparseCore collective offload"
type: experiment
hypothesis: "Enabling concurrent SparseCore collective offload on top of the v210 SparseCore aggregator frontier may reduce async-done tail time and improve step throughput."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-all-collectives-aggregator-concurrent-offload
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, concurrent-offload, collective-aggregator, collectives, all-gather, reduce-scatter, all-reduce, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v211 - concurrent SparseCore collective offload

## Hypothesis under test

**Hypothesis**: v210 confirmed the no-scan SparseCore collective-aggregator
frontier at 59,370-59,372 tok/s / 42.6% MFU. Adding
`--xla_tpu_enable_concurrent_sparse_core_offloading=true` may let SparseCore
collective offloads overlap more effectively and reduce the profile's
async-done tail.

**Mechanism**: Keep the v210 frontier stack fixed, including SparseCore offload
for all-gather, reduce-scatter, and all-reduce plus
`--xla_tpu_enable_sparse_core_collective_aggregator=true`; add only concurrent
SparseCore offloading.

**Predicted signal**: Support requires a clean run with best-worker throughput
above the confirmed v210 band or a profile-level reduction in async-done time
without moving cost elsewhere.

**Falsification criterion**: Clean completion at or below the v210 throughput
band, higher async-done/data-formatting time, increased memory, invalid loss, or
compile/runtime failure.

## Setup

GKE workload `alekseyv-qwen3-v211-noscan-scconc`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v211-no-scan-sparsecore-concurrent-offload`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

Runtime flag delta from v210:

```text
--xla_tpu_enable_concurrent_sparse_core_offloading=true
```

## Equivalence

No model-code edit is made. This is a runtime-flag-only scheduling probe and
preserves model math.

## Results

Completed cleanly with `EXIT_CODE=0` on both workers.

- Worker 0 reported **59,268 tok/s** aggregate, **7,408 tok/s/chip**, and
  **42.5% MFU**.
- Worker 1 reported **59,230 tok/s** aggregate, **7,404 tok/s/chip**, and
  **42.5% MFU**.
- Loss decreased normally through step 19, ending at **12.0459**.
- Worker0 post-profile steps 15-19 were 59,411, 59,338, 59,290, 59,241, and
  59,236 tok/s.

This is below the v210 confirmed best-worker band of 59,370-59,372 tok/s.

## Profile

Profile capture completed on both hosts:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v211-no-scan-sparsecore-concurrent-offload/plugins/profile/2026_06_05_03_04_57/gke-tpu-964065d9-9vh8.{trace.json.gz,xplane.pb}`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v211-no-scan-sparsecore-concurrent-offload/plugins/profile/2026_06_05_03_04_57/gke-tpu-964065d9-c180.{trace.json.gz,xplane.pb}`

XProf run:
`2026-06-05-qwen3-jax-v211-no-scan-sparsecore-concurrent-offload/2026_06_05_03_04_57`.

XProf overview:

- Step time average: **4453.8 ms**, standard deviation **23.2 ms**.
- MXU utilization: **67.2%**.
- TC idle: **6.87 ms** average.
- SparseCore compute: **73.30 ms** average; SparseCore idle:
  **3345.70 ms** average.

Top op-profile buckets across all hosts:

- `convolution fusion`: **57,674.682 ms** / **54.1%**.
- `async-done`: **3,787.366 ms** / **3.5%**.
- `data formatting`: **2,952.746 ms** / **2.8%**.

Compared with v210, the concurrent-offload flag did not reduce the dominant
convolution bucket and slightly increased async-done plus data-formatting time.

## HLO Dump

Fresh train-step HLO was emitted:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v211-no-scan-sparsecore-concurrent-offload/hlo/module_0294.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `ba6e2d7aa46de6265678865f1dc30d9645926b402ed667d8453b15a22ac75a41`
- HLO size: **94,199 lines**, **16,545,183 bytes**.
- Compiled memory: `29723632600` bytes / **27.68 GiB**.

The optimized HLO changed from v210:

| Metric | v210 | v211 |
|---|---:|---:|
| SHA256 | `d67433daa...` | `ba6e2d7a...` |
| HLO bytes | 16,542,528 | 16,545,183 |
| Compiled memory | 27.66 GiB | 27.68 GiB |
| `all-gather` text occurrences | 8,285 | 8,289 |
| `custom-call` text occurrences | 4,019 | 4,025 |
| `fusion` text occurrences | 17,344 | 17,328 |

## Verdict

**Refuted.** Concurrent SparseCore offloading is valid but regresses the
confirmed v210 frontier: lower throughput, slightly higher compiled memory, a
changed optimized HLO, and no profile evidence that async-done tail time moved
in the right direction. Do not carry
`--xla_tpu_enable_concurrent_sparse_core_offloading=true`.
