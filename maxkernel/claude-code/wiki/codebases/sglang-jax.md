---
title: "sglang-jax (sgl-project/sglang-jax)"
type: codebase
tags: [sglang, sglang-jax, pallas, mosaic-tpu, ragged-paged-attention, speculative-decoding, eagle, tree-verify, simple-gla, crown-jewel-tuning-tables]
repo: sgl-project/sglang-jax
commit: 7907875a
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [sglang-jax overview](sglang-jax/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

SGLang's JAX port — mostly a **vendored subset of [tpu-inference](tpu-inference.md)** plus SGLang-specific additions. The two genuinely-novel contributions are **speculative-decoding tree kernels** (EAGLE tree sampling, draft-tree building, greedy tree verify — no public analogue anywhere else) and the **largest tuning table in the ecosystem** (~2,000+ RPA entries across v4 / v5 / v6e / v7). Ingested with the narrow scope the Wave 4 recommendation suggested: speculative-decoding kernels + the tuning table; the vendored kernels are documented but deferred to [tpu-inference](tpu-inference.md). Per-kernel detail in [§3.2 of the Pallas kernel directory](../analyses/pallas-kernel-directory/03-inference-engines.md#32-sgl-projectsglang-jax).

## Architecture

```
python/sgl_jax/srt/kernels/
  ragged_paged_attention/
    ragged_paged_attention.py            ← vendored from tpu-inference v0.11.1
    ragged_paged_attention_v3.py         ← vendored + SGLang extras: custom_mask, attention_sink, xai_temperature
    tuned_block_sizes.py                 ← ~2,000+ entries across v4/v5/v6e/v7  ← LARGEST IN ECOSYSTEM
  paged_attention/
    paged_attention.py                   ← vendored from jax-ml/jax + SPMD extras
  gmm/megablox_gmm_kernel/
    gmm.py, gmm_v2.py, tuned_block_sizes.py       ← vendored from tpu-inference
  fused_moe/v1/
    kernel.py, tuned_block_configs.py            ← vendored from tpu-inference
  quantized_matmul/quantized_matmul_kernels/
    blockwise_kernel.py, kernel.py, tuned_block_sizes.py   ← vendored from tpu-inference
  update_kv_cache/
    update_kv_cache.py, tuned_block_sizes.py     ← vendored (64 MB VMEM budget)
  simple_gla/
    simple_gla.py                        ← vendored from now-private primatrix/pallas-kernel  ← PROVENANCE RISK
  speculative/                           ← NOVEL (no public analogue)
    tree_speculative_sampling_target_only_kernel.py
    build_eagle_tree_structure_kernel.py
    verify_tree_greedy_kernel.py
multimodal/kernels/
  flash_attention.py                     ← vendored directly from jax-ml/jax (bypasses tpu-inference)
  tuned_block_sizes.py, get_block_spec_config.py
```

## Key abstractions

| Kernel | File | Novel? | Notes |
|---|---|---|---|
| `tree_speculative_sampling_target_only` | `kernels/speculative/tree_speculative_sampling_target_only_kernel.py` | **novel** | EAGLE tree sampling; no license header, no docstring |
| `build_eagle_tree_structure` | `kernels/speculative/build_eagle_tree_structure_kernel.py` | **novel** | On-device draft-tree build |
| `verify_tree_greedy` | `kernels/speculative/verify_tree_greedy_kernel.py` | **novel** | Greedy tree verification |
| `ragged_paged_attention_v3` (with custom_mask, attention_sink, xai_temperature, cu_kv_lens) | `kernels/ragged_paged_attention/ragged_paged_attention_v3.py` | **extends** tpu-inference | SGLang additions on top of vendored tpu-inference v0.11.1 |
| RPA tuned block sizes (~2,000+ entries) | `kernels/ragged_paged_attention/tuned_block_sizes.py` | **extends** | v4 small-memory case specially handled; ≥v4 only |
| `simple_gla` (fused_recurrent + chunked) | `kernels/simple_gla/simple_gla.py` | vendored | From `primatrix/pallas-kernel` rev `41431b1, release/v0.4` — **upstream repo went private**; file now carries provenance comment |
| Multimodal `flash_attention` | `multimodal/kernels/flash_attention.py` | vendored | From `jax-ml/jax` directly (not via tpu-inference) |

## Entry points

```bash
cd raw/code/sglang-jax

# install per their docs
pip install -e .

# kernel imports
python -c "from sgl_jax.srt.kernels.speculative.tree_speculative_sampling_target_only_kernel import *"
python -c "from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import *"
```

## Dependencies

- `jax` + `libtpu`.
- Heavier serving / runtime stack.

## Performance-relevant surfaces

### 1. Speculative-decoding tree kernels — only public Pallas
The three files in `kernels/speculative/` are the only public TPU Pallas implementations of EAGLE-style tree verification. If speculative decoding enters any serving hypothesis, they are the reference.

### 2. Largest tuning table in the ecosystem (~2,000+ entries)
`kernels/ragged_paged_attention/tuned_block_sizes.py`. Covers v4 / v5 / v6e / v7. v4 small-memory case is specially handled ("TPU version must be 4 or higher"). **Directly usable prior** for any RPA hypothesis across generations.

### 3. RPA v3 SGLang-specific extensions
`custom_mask` (for spec decoding), `attention_sink` (streaming), `xai_temperature` (Grok), `cu_kv_lens`-based page_indices. Precedent for how to extend an upstream RPA with model-specific knobs without forking.

### 4. Multimodal flash-attention separate table
`multimodal/kernels/flash_attention.py` + its own `tuned_block_sizes.py` + `get_block_spec_config.py`. Cross-attention / vision-encoder flash is distinct enough to warrant its own table — informative if a multimodal hypothesis enters the program.

### 5. Provenance risk: `simple_gla` from now-private upstream
File header: *"Adapted from https://github.com/primatrix/pallas-kernel (rev 41431b1, release/v0.4) — Vendored to remove external dependency after the upstream repository went private."* Consume as-is; do not expect upstream updates. Cite the rev explicitly.

### 6. Vendoring graph — read tpu-inference first
For every non-novel kernel here, the authoritative source is [tpu-inference](tpu-inference.md). Cite sglang-jax only when citing a SGLang-specific extension.

## Connections

- **[Pallas kernel directory §3.2](../analyses/pallas-kernel-directory/03-inference-engines.md#32-sgl-projectsglang-jax)** — full kernel detail + vendoring graph.
- [tpu-inference](tpu-inference.md) — upstream for RPA / fused_moe / megablox / quantized_matmul / update_kv_cache.
- [jax](jax.md) — upstream for multimodal `flash_attention` and paged_attention.

## See also

- [pallas-kernel](../concepts/pallas-kernel.md)
- [continuous-batching](../concepts/continuous-batching.md)
- [kv-cache](../concepts/kv-cache.md)
- [static-cache](../concepts/static-cache.md)
- [attention-block-sizes](../concepts/attention-block-sizes.md)

## Sources

- `raw/code/sglang-jax/python/sgl_jax/srt/kernels/`
- `raw/code/sglang-jax/python/sgl_jax/srt/multimodal/kernels/`
- Upstream: <https://github.com/sgl-project/sglang-jax>
