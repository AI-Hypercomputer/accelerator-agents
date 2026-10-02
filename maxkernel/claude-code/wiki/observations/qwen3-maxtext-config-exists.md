---
title: "MaxText ships a qwen3-8b config matching our model exactly (reference lane is runnable)"
type: observation
tags: [qwen3-cc, maxtext, reference, config]
created: 2026-06-02
updated: 2026-06-02
---

# MaxText ships a qwen3-8b config that matches our model exactly

MaxText defines `configs/models/qwen3-8b.yml` (present in both `raw/code/maxtext` @ `532c8b3d8` and the
`tpu-recipes-v0.1.4` worktree @ `9f1820b47`). Every dimension matches the trainer we optimized on the
jax/torchax lanes — so a **MaxText reference ceiling for this exact architecture is runnable**.

## What was observed

| Param | MaxText `qwen3-8b.yml` | our HF Qwen3-8B trainer |
|---|---|---|
| emb_dim / hidden | 4096 | 4096 |
| query heads | 32 | 32 |
| kv heads | 8 | 8 |
| head_dim | 128 | 128 |
| mlp_dim | 12288 | 12288 |
| decoder layers | **36** | 36 |
| QK-norm | `use_qk_norm: True` | yes |
| activation | SwiGLU (silu) | SwiGLU |
| vocab | 151936 | 151936 |
| rope θ | 1e6 | 1e6 |
| `decoder_block` | `qwen3` | — (custom Flax NNX) |

(Resolves a prior ambiguity: **36 layers**, not 28 — one profile-analyzer sub-report said 28, which was wrong.)

## How to see it

- `raw/code/maxtext/src/maxtext/configs/models/qwen3-8b.yml` (and `qwen3-8b-base.yml`).
- MaxText ships qwen3 configs from 0.6B to 480B-A35B; the dense 8B is `qwen3-8b`.

## Implications

- **No published MFU exists** for MaxText Qwen3-8B: there is no `tuning_params`/throughput target in the
  config and **no tpu-recipes recipe** for qwen3-8b (tpu-recipes' Qwen coverage is the 235B MoE). MaxText
  defines the architecture, not a benchmarked number.
- A reference run is feasible via the same `benchmark_runner` flow the llama3 MaxText baseline used; two
  benchmark configs (`qwen3_8b_8192_ref`, `qwen3_8b_2048_ref`) were added to the worktree mirroring the
  llama3 recipe tuning. Tracked on the [maxtext lane page](../models/qwen3-cc-maxtext.md).
- Expectation by analogy (MaxText Llama-3.1-8B = 44.6% MFU on the same cluster; same architecture class):
  ~40–45% MFU regime — to be measured, not assumed.

## Seen in

- MaxText config inspection, 2026-06-02 (in service of the qwen3_cc maxtext reference lane).
