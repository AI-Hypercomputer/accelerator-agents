---
title: "alphafold3 (google-deepmind/alphafold3) — pinned v3.0.1"
type: codebase
tags: [alphafold3, deepmind, pallas, mosaic-gpu, triton, fused-glu, matmul, autotuning, v3-0-1-pinned]
repo: google-deepmind/alphafold3
commit: 231efc9
pinned_tag: "v3.0.1"
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [alphafold3 overview](alphafold3/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Ingested scope: **only `src/alphafold3/jax/gated_linear_unit/`**, pinned to **tag `v3.0.1`** (commit `231efc9`). The kernels were **removed from `main` after v3.0.1** — always pin the v3.0.1 tag in links. Contains the only public, production-grade **Pallas fused-GLU kernel** (GPU, via Triton-on-Pallas), plus an autotuned block-size search. This is the reference architecture for "what a fused `activation(x @ W_gate) * (x @ W_up)` kernel looks like" — directly informative for closing tokamax's missing TPU Pallas fused GLU, though porting to Mosaic-TPU requires HLO-level validation that XLA isn't already fusing the pattern (same lesson as exp 33's RMSNorm result). Per-kernel detail in [§4.5 of the Pallas kernel directory](../analyses/pallas-kernel-directory/04-research-labs.md#45-google-deepmindalphafold3-pinned-to-tag-v301).

## Overview — scope

Only the GLU directory is in scope. Everything else (model network, training pipeline, data processing) is out of scope for this wiki.

```
src/alphafold3/jax/gated_linear_unit/           ← the ingested subsystem
  gated_linear_unit.py                            ← public API (tries triton, falls back to XLA)
  matmul_ext.py                                   ← PallasGatedLinearUnit + _gated_linear_unit_kernel
  matmul_config.py                                ← autotuned block-size search
  block.py                                        ← masked load_block / store_block / in_bounds_mask
```

Also referenced but not ingest-focus: `src/alphafold3/jax/attention/{flash_attention.py,attention.py}` — a Triton (via `jax_triton`, not Pallas) flash-attention. Noteworthy pattern only; not the reason we ingested.

## Key abstractions

| Kernel | File | Backend | Stability | Perf claims |
|---|---|---|---|---|
| **`PallasGatedLinearUnit`** (`_gated_linear_unit_kernel`) | `src/alphafold3/jax/gated_linear_unit/matmul_ext.py` | `mosaic_gpu` via Triton (Pallas) | stable | "Computes `activation(x @ weight[:, 0]) * x @ weight[:, 1]`" with optional in-kernel epilogue + `dst` aliasing |
| `gated_linear_unit()` public API | `src/alphafold3/jax/gated_linear_unit/gated_linear_unit.py` | dispatcher | stable | Tries triton, falls back to XLA on exception |
| Autotune search | `src/alphafold3/jax/gated_linear_unit/matmul_config.py` | — | stable | Search space: `block_m ∈ [32, 128]`, `block_n ∈ [32, 256]`, `block_k = 32`, split-K fallback when `num_blocks < core_count` |
| `load_block` / `store_block` / `in_bounds_mask` helpers | `src/alphafold3/jax/gated_linear_unit/block.py` | — | reusable | Masked-IO primitives usable in any Pallas kernel |

## Entry points

```bash
cd raw/code/alphafold3
git checkout v3.0.1   # already pinned

# the kernel API
python -c "from alphafold3.jax.gated_linear_unit import gated_linear_unit"
```

## Dependencies

- `jax` + `jax_triton` (Triton-on-Pallas).
- CUDA / A100-H100 runtime (the kernel is GPU; no TPU backend in v3.0.1).

## Performance-relevant surfaces

### 1. Only public production-quality Pallas fused GLU
Every other fused GLU in the ecosystem is either XLA-fallback (tokamax TPU), `triton` FX (tokamax GPU), or forward-only with no custom_vjp ([pallas-forge](pallas-forge.md)). This is the reference. If a TPU-Mosaic port is ever undertaken, this file is the target spec.

### 2. Autotuned block-size search shape
`matmul_config.py`: `block_m ∈ [32, 128]`, `block_n ∈ [32, 256]`, `block_k = 32`, split-K fallback when grid under-fills cores. Directly citable as a starting search space for any GEMM+activation+GEMM kernel.

### 3. Grouped program-IDs for L2 reuse
`_get_best_pids` / `_get_group_cache_usage` in `matmul_ext.py` pick a PID ordering to minimize L2 footprint. **Concept-page candidate** per the directory analysis.

### 4. Optional epilogue + dst-aliasing patterns
Kernel supports optional epilogue (post-activation transform fused in) and output-buffer aliasing. Pattern template for any fused-activation kernel.

### 5. `block.py` masked-IO primitives
`load_block(x, in_bounds_mask)` / `store_block(y, out_val, in_bounds_mask)` — reusable Pallas utilities for handling non-aligned shapes. Apply to any new Pallas kernel.

### 6. Tag-pinning discipline required
**Every reference must include `@v3.0.1`**. Kernels were pulled from `main` after v3.0.1. `.gitmodules` entry tracks HEAD, but the working tree is checked out at the tag.

## Connections

- **[Pallas kernel directory §4.5](../analyses/pallas-kernel-directory/04-research-labs.md#45-google-deepmindalphafold3-pinned-to-tag-v301)** — per-kernel detail.
- [tokamax](tokamax.md) — missing TPU Pallas fused GLU; this is the reference for filling that gap.
- [pallas-forge](pallas-forge.md) — has a forward-only SwiGLU that loses to XLA on v5e; alphafold3 is the "how to do it right" spec.
- Gemma4 exp 33 "Pallas loses when XLA already fuses" lesson — TPU port needs HLO-level validation before committing.
- [sources/2025-ultrascale-playbook.md](../sources/2025-ultrascale-playbook.md) Gap #4 partial retraction.

## See also

- [gated-linear-unit](../concepts/gated-linear-unit.md)
- [xla-fusion](../concepts/xla-fusion.md)
- [pallas-kernel](../concepts/pallas-kernel.md)
- [autotuning](../concepts/autotuning.md)

## Sources

- `raw/code/alphafold3/src/alphafold3/jax/gated_linear_unit/*.py` (at tag v3.0.1)
- Upstream tagged: <https://github.com/google-deepmind/alphafold3/tree/v3.0.1>
- Upstream `main` (no longer has these kernels): <https://github.com/google-deepmind/alphafold3>
