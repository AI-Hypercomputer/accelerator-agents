---
title: "ejkernel (erfanzar/ejkernel)"
type: codebase
tags: [ejkernel, pallas, mosaic-tpu, triton, community, flash-attention, paged-attention, mla, gated-delta-rule, quantized-matmul, all-gather-matmul, ring-attention-wrapper]
repo: erfanzar/ejkernel
commit: f2289a0
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [ejkernel overview](ejkernel/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Single-author community kernel library positioning as "production-grade" — **broadest community TPU Pallas surface** surveyed (17 TPU kernels). Consumed by [EasyDeL](EasyDeL.md) via an operations registry. License Apache-2.0. Each TPU kernel is a folder under `ejkernel/kernels/_pallas/tpu/<kernel>/` with `_interface.py`, `_pallas_impl_fwd.py`, `_pallas_impl_bwd.py`. **Ring-attention here is explicitly a Splash wrapper — orchestration, not a novel kernel.** Per-kernel detail in [§6.1 of the Pallas kernel directory](../analyses/pallas-kernel-directory/06-community-research.md#61-erfanzarejkernel).

## Overview

```
ejkernel/
  kernels/
    _pallas/
      tpu/                                    ← all TPU Pallas kernels live here
        flash_attention/
        flash_mla/
        blocksparse_attention/
        deepseek_attn/
        page_attention/
        prefill_page_attention/
        ragged_page_attention_v2/
        ragged_page_attention_v3/
        ragged_decode_attention/
        multi_latent_ragged_page_attention/
        multi_latent_ragged_page_attention_v2/
        gated_delta_rule/                    ← README claim: 3.6× speedup
        ragged_gated_delta_rule/
        grouped_matmul/
        grouped_matmulv2/
        grouped_matmulv3/
        quantized_matmul/
        all_gather_matmul/
        reduce_scatter_matmul/
        ring_attention/                      ← wraps Splash with ring orchestration
      gpu/                                    ← parallel GPU surface
    _cuda/, _cute/                            ← CUDA + CUTLASS variants
```

## Key abstractions

TPU-side kernels (all `mosaic_tpu`, all experimental per path + docstrings):

| Kernel | Directory | Stage | Notes |
|---|---|---|---|
| `flash_attention` | `_pallas/tpu/flash_attention/` | fwd + bwd | "O(N) memory complexity" |
| `flash_mla` | `_pallas/tpu/flash_mla/` | fwd + bwd | DeepSeek-style MLA |
| `blocksparse_attention` | `_pallas/tpu/blocksparse_attention/` | training | explicit mask module |
| `deepseek_attn` | `_pallas/tpu/deepseek_attn/` | fwd + bwd | DeepSeek-specific path |
| `page_attention`, `prefill_page_attention`, `ragged_page_attention_v2`, `ragged_page_attention_v3` (+ `_h64`) | `_pallas/tpu/*` | fwd-only | Parallels Google's RPA family |
| `ragged_decode_attention` | `_pallas/tpu/ragged_decode_attention/` | decode-only | — |
| `multi_latent_ragged_page_attention` v1/v2 | `_pallas/tpu/*` | fwd-only | MLA paged decode |
| `gated_delta_rule` + `ragged_gated_delta_rule` | `_pallas/tpu/*` | fwd+bwd / fwd | **README: "3.6× speedup"** for ragged GDR decode |
| `grouped_matmul` v1/v2/v3 | `_pallas/tpu/grouped_matmul*/` | fwd + bwd | Three staged iterations in-tree |
| `quantized_matmul` | `_pallas/tpu/quantized_matmul/` | fwd + bwd | Benchmarks vs XLA in `benchmarks/` |
| `all_gather_matmul` | `_pallas/tpu/all_gather_matmul/` | — | **Novel** TPU-native AG∘matmul |
| `reduce_scatter_matmul` | `_pallas/tpu/reduce_scatter_matmul/` | — | Complements above |
| `ring_attention` | `_pallas/tpu/ring_attention/` | — | **Wraps Splash** (not a novel kernel) |

## Entry points

```bash
cd raw/code/ejkernel
pip install -e .

python -c "from ejkernel.kernels._pallas.tpu.flash_attention import flash_attention"
```

## Dependencies

- `jax` + `libtpu`.
- Triton (for `_pallas/gpu/` paths).
- CUDA / CUTLASS (for `_cuda/`, `_cute/`).

## Performance-relevant surfaces

### 1. Broadest community TPU Pallas surface (17 kernels)
If a hypothesis has no first-party kernel (`jax-ml/jax` / `tpu-inference` / `maxtext` / `axlearn`), ejkernel is the next place to look. Trade-off: experimental, single-maintainer.

### 2. Claimed 3.6× ragged GDR decode speedup
From `ejkernel/README.md`. Not verified by an independent microbench in the wiki. Cite as "ejkernel README claim" not measured fact.

### 3. Ring-attention as Splash wrapper (orchestration pattern)
`_pallas/tpu/ring_attention/` reuses Splash's bf16 tuning; adds ring-axis orchestration. Same pattern as [maxdiffusion](maxdiffusion.md)'s `ring_attention_kernel.py` but different maintainer. Lower-stakes copy for anyone wanting to inspect the pattern.

### 4. `_interface.py` / `_pallas_impl_fwd.py` / `_pallas_impl_bwd.py` per-kernel split
Consistent layout across all kernels; each separates dispatch (`_interface.py`) from the Pallas kernel body. Readable pattern for organizing a multi-kernel repo.

### 5. Three staged `grouped_matmul` iterations in-tree
v1 → v2 → v3 sit side-by-side with no removal. Informative for reading as a mini case study of how an MoE kernel evolved.

### 6. Apache-2.0 — freely mixable
No AGPL complication (unlike aphrodite-engine).

## Connections

- **[Pallas kernel directory §6.1](../analyses/pallas-kernel-directory/06-community-research.md#61-erfanzarejkernel)** — full kernel detail.
- [EasyDeL](EasyDeL.md) — consumer/wrapper; registers ejkernel kernels via the operations registry.
- [tpu-inference](tpu-inference.md) / [sglang-jax](sglang-jax.md) — production-grade alternatives for the same kernel families.
- [ringattention](ringattention.md) — canonical paper impl to compare with ejkernel's splash wrapper.

## See also

- [pallas-kernel](../concepts/pallas-kernel.md)
- [splash-attention](../concepts/splash-attention.md)
- [flash-attention](../concepts/flash-attention.md)
- [ragged-dot](../concepts/ragged-dot.md)
- [int8-quantization](../concepts/int8-quantization.md)

## Sources

- `raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/`
- `raw/code/ejkernel/README.md`
- Upstream: <https://github.com/erfanzar/ejkernel>
