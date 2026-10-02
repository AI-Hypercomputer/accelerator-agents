---
title: "maxdiffusion (AI-Hypercomputer/maxdiffusion)"
type: codebase
tags: [maxdiffusion, diffusion, pallas, mosaic-tpu, splash, ring-attention, tokamax-ring, flux, wan, sdxl, ltx]
repo: AI-Hypercomputer/maxdiffusion
commit: c98002fe
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [maxdiffusion overview](maxdiffusion/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Google AI-Hypercomputer's reference JAX diffusion trainer — Flux / Wan / SDXL / LTX. Ingestion narrow-scoped to `src/maxdiffusion/kernels/splash_attention/` per the analysis's Wave 4 scoping recommendation — **the only repo where ring-attention is wired in as a first-class splash-integrated kernel**. Announced 2026-04-16: *"Support for Tokamax Ring Attention kernel is now added."* Per-kernel detail in [§2.2 of the Pallas kernel directory](../analyses/pallas-kernel-directory/02-ai-hypercomputer.md#22-ai-hypercomputermaxdiffusion).

## Overview

All Pallas code lives in one directory:

```
src/maxdiffusion/kernels/splash_attention/
  splash_attention_kernel.py          ← vendored from jax-ml/jax splash; fused bwd + _ring_raw
  ring_attention_kernel.py            ← NEW: splash-integrated ring attention
  splash_attention_mask.py            ← from upstream
  splash_attention_mask_info.py       ← from upstream
  base.py                             ← mask base class
```

The single relevant dispatcher is `src/maxdiffusion/models/attention_flax.py`:

```python
attention_kernel ∈ {"flash", "tokamax_flash", "tokamax_ring"}
```

`"flash"` → upstream splash. `"tokamax_flash"` → repo-local fork. `"tokamax_ring"` → ring-attention kernel below.

## Key abstractions

| Kernel | File | Stability | Notes |
|---|---|---|---|
| `splash_attention_kernel.make_splash_mha` / `make_dynamic_splash_mha` (fwd + **fused bwd** + `_ring_raw`) | `src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py` | stable | Vendored from jax-ml/jax. **Fused bwd only** (block_q_dq / block_kv_dq unused) |
| `ring_attention_kernel.make_ring_attention` | `src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py` | stable (announced 2026-04-16) | Composes with splash (imports `_splash_attention_forward`, `_splash_attention_forward_ring_raw`, `_splash_attention_bwd`). Shifts K/V via `lax.ppermute` across `ring_axis`; online-softmax combine. `rotate_segment_ids`, `is_mqa` options; custom VJP |
| `_flash_attention_dq_kernel` / `_flash_attention_dkv_kernel` | same file | stable | Legacy unfused bwd — MaxDiffusion forces fused bwd for tokamax flavor |

## Entry points

```bash
cd raw/code/maxdiffusion

# install
pip install -e .

# run a sample training
python -m maxdiffusion.train src/maxdiffusion/configs/flux_dev_base.yml \
  attention_kernel=tokamax_ring
```

## Dependencies

- `jax` / `jaxlib` / `libtpu`.
- `tokamax` (optional — for `tokamax_flash` path).
- `flax`, `orbax`.

## Performance-relevant surfaces

### 1. Splash-integrated ring attention (the reference pattern)
`ring_attention_kernel.py` is the **only public TPU reference** for how to wire ring-attention on top of splash (vs. haoliuhl's from-scratch flash inner kernel). API: `make_ring_attention(mask_info, mask, config, ring_axis, rotate_segment_ids, ...)`. Reuses splash fwd + bwd primitives — gets splash's bf16 tuning for free. Directly addresses [sources/2025-ultrascale-playbook.md](../sources/2025-ultrascale-playbook.md) Gap #2.

### 2. Forced fused backward for diffusion
`attention_blocks_flowchart.md`: *"tokamax_flash uses the splash attention implementation in tokamax-repo. This kernel only supports fused backward pass where gradients for q,k,v are computed in a single kernel so block_q_dq and block_kv_dq are not used."* Relevant fact for any hypothesis comparing fused vs unfused bwd on TPU.

### 3. Auto-overwrite of KV block sizes for cross-attention
Per the flowchart: *"cross-attention KV block sizes are auto-overwritten because text-token sequences are shorter than defaults"*. Pattern worth emulating in gemma4 if cross-attention paths are added.

### 4. Long-sequence / high-resolution regime (diffusion)
Frames × resolution / VAE / patchify ratios produce sequences far longer than LLM training. The ring-attention integration is principally motivated by this — informative even for LLM work because the scaling question is the same.

### 5. Shared mask/mask_info with upstream splash
`splash_attention_mask.py` / `splash_attention_mask_info.py` / `base.py` are direct upstream copies. Any mask-class extension upstream is applicable here too.

## Connections

- **[Pallas kernel directory §2.2](../analyses/pallas-kernel-directory/02-ai-hypercomputer.md#22-ai-hypercomputermaxdiffusion)** — per-kernel detail.
- [ringattention](ringattention.md) — paper-companion canonical impl; compare orchestration patterns.
- [ejkernel](ejkernel.md) — also ships a ring-attention splash wrapper; same pattern, different maintainer.
- [tokamax](tokamax.md) — has `ring_attention_kernel` in its experimental tree; maxdiffusion's integration demonstrates how to expose it. Closes Wave 1 reachability gap.
- [jax](jax.md) — upstream splash source.
- [maxtext](maxtext.md) — sibling repo; does NOT ship ring-attention.

## See also

- [ring-attention](../concepts/ring-attention.md)
- [splash-attention](../concepts/splash-attention.md)
- [context-parallelism](../concepts/context-parallelism.md)
- [attention-block-sizes](../concepts/attention-block-sizes.md)

## Sources

- `raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/`
- `raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py`
- Flowchart: `docs/attention_blocks_flowchart.md`
- Upstream: <https://github.com/AI-Hypercomputer/maxdiffusion>
