---
title: "Qwen3 JAX v214 no-scan rematerialized tiled CE bs4 (INVALID, compile OOM: 35.78G HBM)"
type: experiment
tags: [experiment, qwen3-cc, jax, v6e-8, ce, logit-memory, remat, invalid]
hypothesis: qwen3-jax-rematerialized-vocab-tiled-ce
model: qwen3-cc-jax
created: 2026-06-05
updated: 2026-06-05
status: invalid
---

# v214 no-scan rematerialized tiled CE bs4

## Hypothesis Under Test

[v213](2026-06-05-v213-no-scan-tiled-ce-bs4.md) proved the naive Python/XLA
vocab-tiled CE implementation is invalid: it removes the full-vocab logits but
autodiff saves many per-chunk logits and masks, causing a worse compile OOM.
This experiment kept the v213 trainer and runtime stack fixed, but wrapped the
per-chunk max and sum/picked-logit computations in `jax.checkpoint` so the
backward pass would preferentially rematerialize chunk work instead of saving
chunk logits and predicates.

Expected outcome:

- Supported if rematerialized chunks bring bs4/chip train-step HBM back below
  the v6e-8 31.25 GiB limit.
- Invalid if chunk logits/predicates remain live enough to OOM before step 0.

## Setup

- **Workload**: `alekseyv-qwen3-v214-remtiled-bs4`
- **Cluster**: `<your-cluster>` (`<your-region>`, project
  `<your-project>`, v6e-8, 2 hosts x 4 chips)
- **Image**:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v214-remat-tiled-ce`
- **Image push digest**:
  `sha256:6b31c89295cb9f207bfe2ac25b66b8660e29d72180e355d9ea08e9bc4c58fb0c`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v214-no-scan-remat-tiled-ce-bs4`
- **Base stack**: v210 no-scan, remat, Tokamax Splash, MaxText CE disabled,
  SparseCore offload for all-gather/reduce-scatter/all-reduce plus aggregator,
  collective-matmul modes `none`, scoped VMEM 100352 KiB.
- **Shape**: seq8192, per-chip batch 4, global batch 32.
- **CE flags**: `--use_tiled_ce=True --tiled_ce_chunk_size=4096
  --use_maxtext_ce=False`.

## Code Change

The v214 fork changes only `_ce_tiled` relative to v213:

- Adds a small `_chunk_logits` helper for chunk-local lm-head matmul.
- Wraps the first-pass chunk max in `jax.checkpoint`.
- Wraps the second-pass chunk `sum_exp` and selected-label contribution in
  `jax.checkpoint`, returning only `[B*L]` accumulators.

Cheap pre-run validation passed in the image:

- `python -m py_compile /tmp/train.py`
- Toy CPU JAX comparison against full CE:
  - loss delta: `0.0`
  - hidden-grad max delta: `0.00390625`
  - lm-head-grad max delta: `0.0`

## Result

Invalid. The workload failed before step 0 during train-step compilation:

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 35.78G of 31.25G hbm.
Exceeded hbm capacity by 4.54G.
```

No runtime profile exists.

Rematerialization helped, but not nearly enough:

| Metric | v213 naive tiled CE | v214 remat tiled CE | Delta |
|---|---:|---:|---:|
| Program HBM requirement | 36.99 GiB | 35.78 GiB | -1.21 GiB |
| Memory-report total | 42.44 GiB | 41.23 GiB | -1.21 GiB |
| Preallocated temp allocation | 36.69 GiB | 35.48 GiB | -1.21 GiB |
| Optimized train-step HLO | 116,128 lines | 115,286 lines | -842 lines |

The memory report still contains many chunk-shaped values:

```text
78 bf16[32768,4096]
15 f32[32768,4096]
37 pred[32768,4096]
```

The top OOM allocation remains a 512 MiB `f32[32768,4096]` fusion from the
checkpointed lm-head-dot path, followed by many 256 MiB chunk tensors. The
checkpoint annotations reduce the live set modestly but do not turn this into
a streaming CE.

## HLO Artifacts

- Prefix:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v214-no-scan-remat-tiled-ce-bs4/hlo/xla-dump/`
- Recursive object summary: **1220 objects, 122.93 MiB**.
- Train-step optimized HLO:
  `module_0294.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step optimized HLO SHA256:
  `07adb0643ddb20b0f329f1d68a2dc05e32e1c40b28ccf1ce31d025f2d334976f`
- Memory report SHA256:
  `9af31d997e9b5f715a3c45d481bc9a665463866e9059ed4548857adbfea03386`
- OOM top-instructions SHA256:
  `167be2698d1efcc498e3c76b129470aa22eaae698a8b04963953b42dfe08d8dc`

## Verdict

Invalid. Do not carry rematerialized chunked CE at chunk size 4096. The result
is informative because it lowers HBM by about 1.2 GiB, so a smaller chunk-size
probe is worth one cheap compile attempt. If chunk-size scaling is weak, stop
the chunk sweep and move to a custom VJP or Pallas/streaming CE implementation
that explicitly controls the backward live set.
