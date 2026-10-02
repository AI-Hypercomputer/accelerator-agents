---
title: "Qwen3 JAX v216 no-scan custom-VJP tiled CE bs4 (INVALID, compile OOM: 38.12G HBM)"
type: experiment
tags: [experiment, qwen3-cc, jax, v6e-8, ce, logit-memory, custom-vjp, invalid]
hypothesis: qwen3-jax-custom-vjp-vocab-tiled-ce
model: qwen3-cc-jax
created: 2026-06-05
updated: 2026-06-05
status: invalid
---

# v216 no-scan custom-VJP tiled CE bs4

## Hypothesis Under Test

[v213](2026-06-05-v213-no-scan-tiled-ce-bs4.md) and
[v214](2026-06-05-v214-no-scan-remat-tiled-ce-bs4.md) showed that autodiff and
remat-only autodiff over the Python vocab-chunk loop retain too many per-chunk
logits/predicates. [v215](2026-06-05-v215-no-scan-remat-tiled-ce-chunk2048-bs4.md)
closed the simple chunk-size sweep. This experiment kept the forward tiled CE
math and chunk size 4096, but replaced autodiff through the chunk loop with a
custom VJP that explicitly recomputes chunk logits in backward and forms
`softmax - onehot` chunk gradients.

Expected outcome:

- Supported if explicit backward removes the saved chunk masks/logits enough to
  compile under 31.25 GiB.
- Invalid if the unrolled custom backward still leaves many full
  `f32[32768,4096]`/chunk temps live and OOMs.

## Setup

- **Workload**: `alekseyv-qwen3-v216-cvjpce-bs4`
- **Cluster**: `<your-cluster>` (`<your-region>`, project
  `<your-project>`, v6e-8, 2 hosts x 4 chips)
- **Image**:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v216-custom-vjp-tiled-ce`
- **Image digest**:
  `sha256:27bb89cabebf4ea0aeea20733e45ee32be22acbf664e8d84e471e8fc7bc724b5`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v216-no-scan-custom-vjp-tiled-ce-bs4`
- **Base stack**: v210 no-scan, remat, Tokamax Splash, MaxText CE disabled,
  SparseCore offload for all-gather/reduce-scatter/all-reduce plus aggregator,
  collective-matmul modes `none`, scoped VMEM 100352 KiB.
- **Shape**: seq8192, per-chip batch 4, global batch 32.
- **CE flags**: `--use_tiled_ce=True --tiled_ce_chunk_size=4096
  --use_maxtext_ce=False`.

## Code Change

The v216 fork changes only `_ce_tiled` relative to v214:

- Keeps the two-pass stable forward over vocab chunks.
- Defines a nested `jax.custom_vjp` for the scalar CE.
- Stores only hidden, labels, lm-head weight, per-token max logit, and per-token
  `sum_exp` in the VJP residual.
- Recomputes each chunk in backward, subtracts the selected target with scatter,
  accumulates hidden gradients, and concatenates per-chunk lm-head gradients.

Cheap pre-run validation passed:

- `python -m py_compile /tmp/train.py`
- Toy CPU JAX comparison against full CE:
  - loss delta: `0.0`
  - hidden-grad max delta: `0.00146484375`
  - lm-head-grad max delta: `0.0009765625`

## Result

Invalid. The workload failed before step 0 during train-step compilation:

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 38.12G of 31.25G hbm.
Exceeded hbm capacity by 6.87G.
```

No runtime profile exists.

The explicit backward removes the `pred[32768,4096]` live set, but the unrolled
Python custom VJP still expands into many chunk-sized dot products and f32 temps:

| Metric | v214 remat tiled CE | v216 custom-VJP tiled CE |
|---|---:|---:|
| Program HBM requirement | 35.78 GiB | 38.12 GiB |
| Memory-report total | 41.23 GiB | 43.59 GiB |
| Preallocated temp allocation | 35.48 GiB | 37.83 GiB |
| Optimized train-step HLO | 115,286 lines | 109,076 lines |
| `bf16[32768,4096]` in memory report | 78 | 42 |
| `f32[32768,4096]` in memory report | 15 | 52 |
| `pred[32768,4096]` in memory report | 37 | 0 |

The top OOM allocations are 512 MiB `f32[32768,4096]` fusions from
`transpose(forward_backward)/jvp()/dot_general`. The custom VJP fixed one
symptom (saved predicates) but worsened the f32 chunk-dot live set.

## HLO Artifacts

- Prefix:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v216-no-scan-custom-vjp-tiled-ce-bs4/hlo/xla-dump/`
- Recursive object summary: **4950 objects, 1,760,258,711 bytes**.
- Train-step optimized HLO:
  `module_0294.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step optimized HLO size: **109,076 lines / 18,668,321 bytes**.
- Train-step optimized HLO SHA256:
  `4b4277d8e930d9dc40c491334ffd8bffaec38ba4f2b33d9aa4bed537c1d6d8f6`
- Memory report SHA256:
  `05451e8a7e17cd8353407384f15dff7e4cdc017db59f316a2b20b7f75c3e8c3b`
- OOM top-instructions SHA256:
  `0d6efecb0649ed07147591950fca325942a629b0fd253d632426c6dc8375e3a8`

## Verdict

Invalid. Do not carry the unrolled Python custom VJP. The result suggests the
next cheap structural probe is the same custom VJP expressed with `lax.scan`
over vocab chunks, to avoid unrolling 38 backward chunk-dot blocks. If scan
still leaves the f32 chunk-dot live set above capacity, move to a real
Pallas/streaming CE kernel.
