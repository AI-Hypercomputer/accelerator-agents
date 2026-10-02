---
title: "Qwen3 JAX v213 no-scan tiled CE bs4 (INVALID, compile OOM: 36.99G HBM)"
type: experiment
tags: [experiment, qwen3-cc, jax, v6e-8, ce, logit-memory, invalid]
hypothesis: qwen3-jax-vocab-tiled-ce
model: qwen3-cc-jax
created: 2026-06-05
updated: 2026-06-05
status: invalid
---

# v213 no-scan tiled CE bs4

## Hypothesis Under Test

The v210/v212 no-scan SparseCore-aggregator stack is faster than MaxText at the
same bs3/chip shape, but bs5 direct scaling with MaxText CE is still a near-fit
compile OOM. This experiment tested a native-JAX vocab-tiled lm-head CE path at
bs4/chip before attempting bs5. The intended mechanism was to remove the full
`[B,L,V]` logit tensor by computing cross entropy over vocab chunks of 4096.

Expected outcome:

- Supported if tiled CE compiles at bs4 with lower train-step HBM than the
  v210 MaxText CE frontier while preserving CPU equivalence.
- Invalid if XLA autodiff retains chunk intermediates/masks and raises memory
  instead of lowering it.

## Setup

- **Workload**: `alekseyv-qwen3-v213-tiledce-bs4`
- **Cluster**: `<your-cluster>` (`<your-region>`, project
  `<your-project>`, v6e-8, 2 hosts x 4 chips)
- **Image**:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v213-tiled-ce`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v213-no-scan-tiled-ce-bs4`
- **Base stack**: v210 no-scan, remat, Tokamax Splash, MaxText CE disabled,
  SparseCore offload for all-gather/reduce-scatter/all-reduce plus aggregator,
  collective-matmul modes `none`, scoped VMEM 100352 KiB.
- **Shape**: seq8192, per-chip batch 4, global batch 32.
- **New flags**: `--use_tiled_ce=True --tiled_ce_chunk_size=4096
  --use_maxtext_ce=False`.

## Code Change

The v213 fork adds `_ce_tiled(hidden, labels, lm_head_w)` to the trainer. It
flattens hidden states to `[B*L,H]`, loops over `lm_head` vocab chunks, computes
a first-pass global max logit, then computes `sum_exp` and the selected label
logit per chunk.

Cheap pre-run validation passed in the image:

- `python -m py_compile /app/trainer/train.py`
- Toy CPU JAX comparison against full CE:
  - loss delta: `2.384185791015625e-07`
  - hidden-grad max delta: `0.001953125`
  - lm-head-grad max delta: `0.0`

## Result

Invalid. The workload failed before step 0 during train-step compilation:

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 36.99G of 31.25G hbm.
Exceeded hbm capacity by 5.74G.
```

No runtime profile exists.

The HLO memory report is worse than the v210/v212 MaxText CE frontier:

| Metric | v213 tiled CE |
|---|---:|
| Program HBM requirement | 36.99 GiB |
| Memory-report total | 42.44 GiB |
| Preallocated temp allocation | 36.69 GiB |
| Optimized train-step HLO | 116,128 lines / 19,780,378 bytes |

By comparison, v210/v212 MaxText CE compiled around 27.66/22.90 GiB at bs4/bs3
depending on shape, and v202 Tokamax CE bs5 compiled at 25.62 GiB but was too
slow.

The mechanism failed because the plain Python/XLA chunk loop removed the full
`bf16[4,8192,151936]` logits tensor but caused autodiff to retain many chunk
intermediates and masks. The memory report includes:

```text
83 bf16[32768,4096]
10 f32[32768,4096]
38 pred[32768,4096]
```

The top OOM allocation was a 512 MiB `f32[32768,4096]` fusion from the
lm-head-dot backward path, followed by many 256 MiB `bf16[4,8192,4096]`
forward/backward temps.

## HLO Artifacts

- Prefix:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v213-no-scan-tiled-ce-bs4/hlo/`
- Recursive object summary: **903 objects, 116.14 MiB**.
- Train-step optimized HLO:
  `module_0294.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step optimized HLO SHA256:
  `c645d622e27b78b2e72a2130aff9383274a02a06150f96ae3a246c44dcbf47dd`
- Memory report SHA256:
  `d2c180d5305087879c19dc2ee76df5401908b35e17d2caa695f83ab20ff9edb0`
- OOM top-instructions SHA256:
  `2ae301563b5bf3eaf4c551668a16b0d14149325ccbe154396f064244fd035772`

## Verdict

Invalid. Do not carry naive autodiffed tiled CE. A future CE/logit-memory path
must avoid saving per-chunk logits and predicates in the backward pass. The next
reasonable variants are a rematerialized chunk CE as a cheap test, then a custom
VJP or Pallas/streaming CE if remat still leaves chunk masks/residuals live.
