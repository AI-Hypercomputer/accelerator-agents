---
slug: tokamax
commit: 946ae7894132b1b6e06421616f98f49008b215aa
scip_tool: scip-python
updated: 2026-07-03
---

# tokamax internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| tokamax-_src-benchmarking | [tokamax-_src-benchmarking](concepts/tokamax-_src-benchmarking.md) | fresh |
| tokamax-_src-ops-attention-base | [tokamax-_src-ops-attention-base](concepts/tokamax-_src-ops-attention-base.md) | fresh |
| tokamax-_src-ops-attention-pallas_mosaic_gpu_common | [tokamax-_src-ops-attention-pallas_mosaic_gpu_common](concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_common.md) | fresh |
| tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100 | [tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100](concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100.md) | fresh |
| tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100 | [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md) | fresh |
| tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm90 | [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm90](concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm90.md) | fresh |
| tokamax-_src-ops-attention-pallas_triton | [tokamax-_src-ops-attention-pallas_triton](concepts/tokamax-_src-ops-attention-pallas_triton.md) | fresh |
| tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel | [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md) | fresh |
| tokamax-_src-ops-experimental-tpu-splash_attention-ring_attention_kernel | [tokamax-_src-ops-experimental-tpu-splash_attention-ring_attention_kernel](concepts/tokamax-_src-ops-experimental-tpu-splash_attention-ring_attention_kernel.md) | fresh |
| tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel | [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md) | fresh |
| tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test | [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test](concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test.md) | fresh |
| tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask | [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md) | fresh |
| tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask_info | [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask_info](concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask_info.md) | fresh |
| tokamax-_src-ops-flex_attention-pallas_triton | [tokamax-_src-ops-flex_attention-pallas_triton](concepts/tokamax-_src-ops-flex_attention-pallas_triton.md) | fresh |
| tokamax-_src-ops-gated_linear_unit-base | [tokamax-_src-ops-gated_linear_unit-base](concepts/tokamax-_src-ops-gated_linear_unit-base.md) | fresh |
| tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common | [tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common](concepts/tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common.md) | fresh |
| tokamax-_src-ops-op | [tokamax-_src-ops-op](concepts/tokamax-_src-ops-op.md) | fresh |
| tokamax-_src-ops-ragged_dot-base | [tokamax-_src-ops-ragged_dot-base](concepts/tokamax-_src-ops-ragged_dot-base.md) | fresh |
| tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common | [tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common](concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common.md) | fresh |
| tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu | [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md) | fresh |
| tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel | [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md) | fresh |
| tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel | [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel](concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel.md) | fresh |
| tokamax-_src-ops-ragged_dot-pallas_triton | [tokamax-_src-ops-ragged_dot-pallas_triton](concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md) | fresh |
| tokamax-_src-pallas-block | [tokamax-_src-pallas-block](concepts/tokamax-_src-pallas-block.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **2636** across 149 modules
- deep (concept pages): **196** (7.4%)
- catalog-only: **2440**
- represented total: **2636** (100.0%)
- classes represented: **203/203**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
