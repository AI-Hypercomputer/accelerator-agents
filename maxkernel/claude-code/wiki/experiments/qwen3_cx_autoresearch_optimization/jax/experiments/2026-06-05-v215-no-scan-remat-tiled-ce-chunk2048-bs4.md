---
title: "Qwen3 JAX v215 no-scan rematerialized tiled CE chunk2048 bs4 (INVALID, compile OOM: 54.36G HBM)"
type: experiment
tags: [experiment, qwen3-cc, jax, v6e-8, ce, logit-memory, remat, invalid]
hypothesis: qwen3-jax-rematerialized-vocab-tiled-ce-chunk-size
model: qwen3-cc-jax
created: 2026-06-05
updated: 2026-06-05
status: invalid
---

# v215 no-scan rematerialized tiled CE chunk2048 bs4

## Hypothesis Under Test

[v214](2026-06-05-v214-no-scan-remat-tiled-ce-bs4.md) showed that
checkpointing the per-chunk tiled-CE computations reduced the compile OOM by
about 1.2 GiB versus [v213](2026-06-05-v213-no-scan-tiled-ce-bs4.md), but still
missed v6e-8 HBM by 4.54G. This experiment kept the v214 image and runtime
stack fixed and changed only `--tiled_ce_chunk_size=4096` to
`--tiled_ce_chunk_size=2048`, testing whether smaller chunks reduce the
checkpointed lm-head live set enough to fit.

Expected outcome:

- Supported if the smaller chunk size materially lowers train-step HBM and
  compiles under 31.25 GiB.
- Invalid if chunk count/remat scheduling dominates and HBM increases or still
  misses capacity by a wide margin.

## Setup

- **Workload**: `alekseyv-qwen3-v215-remt2k-bs4`
- **Cluster**: `<your-cluster>` (`<your-region>`, project
  `<your-project>`, v6e-8, 2 hosts x 4 chips)
- **Image**:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v214-remat-tiled-ce`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v215-no-scan-remat-tiled-ce-chunk2048-bs4`
- **Base stack**: v210 no-scan, remat, Tokamax Splash, MaxText CE disabled,
  SparseCore offload for all-gather/reduce-scatter/all-reduce plus aggregator,
  collective-matmul modes `none`, scoped VMEM 100352 KiB.
- **Shape**: seq8192, per-chip batch 4, global batch 32.
- **CE flags**: `--use_tiled_ce=True --tiled_ce_chunk_size=2048
  --use_maxtext_ce=False`.

## Result

Invalid. The workload failed before step 0 during train-step compilation:

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 54.36G of 31.25G hbm.
Exceeded hbm capacity by 23.11G.
```

No runtime profile exists.

The smaller chunk size made memory much worse:

| Metric | v213 chunk4096 | v214 remat chunk4096 | v215 remat chunk2048 |
|---|---:|---:|---:|
| Program HBM requirement | 36.99 GiB | 35.78 GiB | 54.36 GiB |
| Memory-report total | 42.44 GiB | 41.23 GiB | 59.76 GiB |
| Preallocated temp allocation | 36.69 GiB | 35.48 GiB | 54.01 GiB |
| Optimized train-step HLO | 116,128 lines | 115,286 lines | 141,911 lines |

The memory report shows both new 2048-wide chunk state and many remaining
4096-wide lm-head/backward values:

```text
132 bf16[32768,2048]
29 f32[32768,2048]
74 pred[32768,2048]
150 bf16[32768,4096]
1 f32[32768,4096]
0 pred[32768,4096]
```

The top allocation is still a 512 MiB `f32[32768,4096]` checkpointed
`dot_general`, followed by many 256 MiB `bf16[32768,4096]` temps. This is a
nonlinear worsening from chunk-count expansion, not a useful path toward a
fit.

## HLO Artifacts

- Prefix:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v215-no-scan-remat-tiled-ce-chunk2048-bs4/hlo/xla-dump/`
- Recursive object summary: **1226 objects, 149,773,227 bytes**.
- Train-step optimized HLO:
  `module_0294.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step optimized HLO size: **141,911 lines / 23,370,047 bytes**.
- Train-step optimized HLO SHA256:
  `4c5b3059980dcc98463a686c10a300c6d939f92c99d576d4710280dab036adb6`
- Memory report SHA256:
  `a2174f7e56b16b35a5011f9f71eb556ccf643884697725b71c802b3975000a6e`
- OOM top-instructions SHA256:
  `9f9d2bb6673ad06e3ac3f5933be91605b1b905e29a1423d4d5dce4c70f6b7081`

## Verdict

Invalid. Do not continue the remat-only chunk-size sweep. Chunk size 2048
increases program HBM from 35.78G to 54.36G, so chunk size 1024 is unlikely to
be productive. The CE/logit-memory branch should move to a custom VJP or
Pallas/streaming CE that controls the backward live set explicitly.
