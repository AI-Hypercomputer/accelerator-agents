---
title: "maxtext (AI-Hypercomputer/maxtext)"
type: codebase
tags: [maxtext, pallas, mosaic-tpu, splash, paged-attention, megablox, sparsecore, gemma, llama, deepseek, trainer]
repo: AI-Hypercomputer/maxtext
commit: 532c8b3d8
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [maxtext overview](maxtext/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Google AI-Hypercomputer's reference JAX trainer for Gemma / Llama / DeepSeek / Qwen / Mistral / Kimi on TPU. **The closest public analogue of this wiki's gemma4 program**. Ships Pallas kernels for training attention (splash), serving attention (paged + ragged), and MoE (megablox GMM with int4/int8 via qwix). Notable: `sc_gather_reduce.py` uses raw MLIR dialects below `pallas_call` for SparseCore (v5p/v7x) — unique MLIR-Pallas entry in the ecosystem. Per-kernel detail in [§2.1 of the Pallas kernel directory](../analyses/pallas-kernel-directory/02-ai-hypercomputer.md#21-ai-hypercomputermaxtext).

## Overview

Attention-kernel selection goes through a config knob: `attention: autoselected | dot_product | flash | cudnn_flash_te | paged | vllm_rpa`. That decision table is itself reference material for how to wire a Pallas kernel into a real trainer.

## Architecture

```
src/maxtext/
  kernels/
    attention/
      splash_attention_kernel.py         ← vendored from jax-ml/jax splash; BlockSizes, ChunkedCausalMask
      ragged_attention.py                 ← ragged_mha / ragged_gqa / ragged_mqa (decode)
      jax_flash_attention.py              ← XLA fallback (not Pallas)
    megablox/
      backend.py                          ← _gmm_fwd / _gmm_bwd (int4/int8 via qwix.pallas QArray)
      ops.py                              ← wrapper; can fall through to tokamax.ragged_dot
      common.py                           ← make_group_metadata
    gather_reduce_sc.py                   ← SparseCore MoE unroute; **MLIR-dialect Pallas (unusual)**
    sort_activations.py                   ← route / unroute / _sort_activations
  inference/
    paged_attention_kernel_v2.py          ← vendored from jax-ml/jax ragged_paged_attention
    paged_attention.py                    ← Linen/NNX wrapper
  layers/
    attention_op.py                       ← config.attention dispatch
    attentions.py                         ← layer wiring (including paged / vllm_rpa hooks)
    moe.py                                ← megablox / tokamax selection
  configs/
    models/                               ← Gemma/Llama/DeepSeek/Qwen/Mistral/Kimi families
    inference/                            ← llama3_405b, llama3_70b, etc.
```

## Key abstractions

Per the [directory §2.1 table](../analyses/pallas-kernel-directory/02-ai-hypercomputer.md#21-ai-hypercomputermaxtext), highlights:

| Kernel | File | Stability | Notes |
|---|---|---|---|
| `splash_attention_kernel.make_splash_mha` (fwd + `_dq` + `_dkv` bwd) | `src/maxtext/kernels/attention/splash_attention_kernel.py` | stable | **Vendored from jax-ml/jax**; `config.use_splash_scheduler` toggles tokamax's extended splash |
| `ChunkedCausalMask` | `src/maxtext/layers/attention_op.py` | stable | For Gemma4 local-sliding/chunk — subclasses `splash_attention_mask._ComputableMask` |
| `ragged_mha` / `ragged_gqa` / `ragged_mqa` | `src/maxtext/kernels/attention/ragged_attention.py` | stable (decode) | `pl.CostEstimate` hints; `use_ragged_attention + ragged_block_size` (default 256) |
| `gmm` / `tgmm` (`_gmm_fwd` / `_gmm_bwd`, int4/int8) | `src/maxtext/kernels/megablox/backend.py` | stable | `qwix.pallas` `QArray`-aware; `use_tokamax_backend=True` falls through to `tokamax.ragged_dot(implementation="mosaic")` |
| `sc_gather_reduce` (SparseCore MoE reduce) | `src/maxtext/kernels/gather_reduce_sc.py` | stable | **Lowered via `jax.experimental.mosaic` dialects** (`arith/func/memref/scf/vector`) — not `pallas_call` |
| `route` / `unroute` / `_sort_activations` | `src/maxtext/kernels/sort_activations.py` | stable | Dispatches to `sc_gather_reduce` via `use_gather_mosaic_kernel: True` |
| `ragged_paged_attention` v2 (vendored) | `src/maxtext/inference/paged_attention_kernel_v2.py` | stable | Code comment: "copied version of jax-ml/jax ragged_paged_attention.py. Clean up when Maxtext migrates" |
| `attention == "vllm_rpa"` | `src/maxtext/layers/attentions.py` | stable | Delegates to [tpu-inference](tpu-inference.md) via `sharded_ragged_paged_attention` |

## Entry points

```bash
cd raw/code/maxtext

# install
pip install -e .

# a training run (per their docs)
python -m maxtext.train src/maxtext/configs/base.yml \
  model_name=llama3-8b attention=autoselected

# serving
python -m maxtext.inference src/maxtext/configs/inference/llama3_70b_v5e-16.yml
```

Perf guide: [`docs/guides/optimization/pallas_kernels_performance.md`](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/pallas_kernels_performance.md).

## Dependencies

- `jax` + `jaxlib` + `libtpu` (TPU).
- `tokamax` (optional — selected by `use_tokamax_backend`, `use_splash_scheduler`, `use_tokamax_gmm`).
- `qwix` (quantization-aware `pallas_call` wrappers for int4/int8 GMM).
- `tpu-inference` (optional — `attention=vllm_rpa`).
- `flax`, `orbax`, heavy trainer deps.

## Performance-relevant surfaces

### 1. `attention:` config-knob dispatch
`src/maxtext/layers/attention_op.py`. Directly readable as a decision tree for how to pick an attention kernel per (model, hardware, regime). Applicable template for this wiki's training loop.

### 2. `BlockSizes` autotune surface
Splash kernel exposes `BlockSizes(block_q, block_kv, block_kv_compute, block_q_dkv, block_kv_dkv, block_kv_dkv_compute, block_q_dq, block_kv_dq, use_fused_bwd_kernel, QKVLayout)` — same structure gemma4 [exp 24](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp24-splash-seq-minor-accepted.md) / [exp 25](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp25-splash-block1024-accepted.md) tune. MaxText uses all eight; gemma4 currently exercises a subset.

### 3. Megablox fall-through to tokamax
`megablox/ops.py` `use_tokamax_backend=True` routes to `tokamax.ragged_dot(implementation="mosaic")`. Precedent for any gemma4 MoE hypothesis.

### 4. SparseCore MLIR-dialect Pallas
`gather_reduce_sc.py` drops below `pallas_call` to `jax.experimental.mosaic` + `jaxlib.mlir.dialects.{arith,func,memref,scf,vector}`. When `pl.pallas_call` doesn't expose the TPU feature you need, this is the template. Autotune surface is large: `col_chunk_size`, `row_chunk_size`, `loop_unroll_factor_{1,2,3}`, `loop_parallel_access_{1,2,3}`, `reduce_group_size`, `single_sc`, `topk_wgt_zero_nan`.

### 5. Inference config-set as hardware-shape reference
`src/maxtext/configs/inference/{llama3_405b_v6e-16-16,llama2_70b_v5e-16,llama3_70b_v5e-16,llama3_405b_v5e-64}.yml` — what Google actually runs. Cite shape-to-hardware mappings from here rather than re-derive.

### 6. Splash + tokamax + vllm_rpa three-way attention integration
Same trainer, three kernel providers. Template for any future hypothesis that compares kernel sources head-to-head in a real trainer.

### 7. Perf-guide document is a first-party recipe
`docs/guides/optimization/pallas_kernels_performance.md` — reference for when to reach for Pallas in a trainer. Worth ingesting as a `source/` page eventually.

## Connections

- **[Pallas kernel directory §2.1](../analyses/pallas-kernel-directory/02-ai-hypercomputer.md#21-ai-hypercomputermaxtext)** — per-kernel detail.
- [jax](jax.md) — upstream splash and ragged_paged_attention.
- [tokamax](tokamax.md) — optional fall-through for splash scheduler + ragged_dot.
- [tpu-inference](tpu-inference.md) — `attention=vllm_rpa` delegates here.
- [maxdiffusion](maxdiffusion.md) — sibling repo; shares splash pattern.
- gemma4 program — directly analogous; this repo is the public reference-point for our trainer.

## See also

- [splash-attention](../concepts/splash-attention.md)
- [attention-block-sizes](../concepts/attention-block-sizes.md)
- [ragged-dot](../concepts/ragged-dot.md)
- [kv-cache](../concepts/kv-cache.md)
- [sparsecore](../concepts/sparsecore.md)
- [expert-parallelism](../concepts/expert-parallelism.md)
- [custom-call](../concepts/custom-call.md)

## Sources

- `raw/code/maxtext/src/maxtext/kernels/`
- `raw/code/maxtext/src/maxtext/inference/`
- `raw/code/maxtext/src/maxtext/layers/`
- `raw/code/maxtext/src/maxtext/configs/`
- Perf guide: `docs/guides/optimization/pallas_kernels_performance.md`
- Upstream: <https://github.com/AI-Hypercomputer/maxtext>
