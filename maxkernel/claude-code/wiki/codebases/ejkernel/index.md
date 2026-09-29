---
slug: ejkernel
commit: f1b5eb128fb82df540d7c38a0f01ea271eb6c79f
scip_tool: scip-python
updated: 2026-07-03
---

# ejkernel internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| ejkernel-callib-_triton_call | [ejkernel-callib-_triton_call](concepts/ejkernel-callib-_triton_call.md) | fresh |
| ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info | [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md) | fresh |
| ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel | [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md) | fresh |
| ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks | [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md) | fresh |
| ejkernel-kernels-_pallas-tpu-flash_attention-_utils | [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md) | fresh |
| ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl | [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md) | fresh |
| ejkernel-kernels-_pallas-tpu-multi_latent_ragged_page_attention_v2-_pallas_impl_fwd | [ejkernel-kernels-_pallas-tpu-multi_latent_ragged_page_attention_v2-_pallas_impl_fwd](concepts/ejkernel-kernels-_pallas-tpu-multi_latent_ragged_page_attention_v2-_pallas_impl_fwd.md) | fresh |
| ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core | [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md) | fresh |
| ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd | [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md) | fresh |
| ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils | [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md) | fresh |
| ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd | [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md) | fresh |
| ejkernel-kernels-_registry | [ejkernel-kernels-_registry](concepts/ejkernel-kernels-_registry.md) | fresh |
| ejkernel-modules-base | [ejkernel-modules-base](concepts/ejkernel-modules-base.md) | fresh |
| ejkernel-modules-operations-configs | [ejkernel-modules-operations-configs](concepts/ejkernel-modules-operations-configs.md) | fresh |
| ejkernel-modules-operations-quantized_matmul | [ejkernel-modules-operations-quantized_matmul](concepts/ejkernel-modules-operations-quantized_matmul.md) | fresh |
| ejkernel-ops-config-selection | [ejkernel-ops-config-selection](concepts/ejkernel-ops-config-selection.md) | fresh |
| ejkernel-ops-core-kernel | [ejkernel-ops-core-kernel](concepts/ejkernel-ops-core-kernel.md) | fresh |
| ejkernel-ops-execution-executor | [ejkernel-ops-execution-executor](concepts/ejkernel-ops-execution-executor.md) | fresh |
| ejkernel-ops-execution-tuning | [ejkernel-ops-execution-tuning](concepts/ejkernel-ops-execution-tuning.md) | fresh |
| ejkernel-ops-utils-datacarrier | [ejkernel-ops-utils-datacarrier](concepts/ejkernel-ops-utils-datacarrier.md) | fresh |
| ejkernel-quantization-_quants-quantizations | [ejkernel-quantization-_quants-quantizations](concepts/ejkernel-quantization-_quants-quantizations.md) | fresh |
| ejkernel-quantization-_utils-qparams | [ejkernel-quantization-_utils-qparams](concepts/ejkernel-quantization-_utils-qparams.md) | fresh |
| ejkernel-quantization-quantized_array | [ejkernel-quantization-quantized_array](concepts/ejkernel-quantization-quantized_array.md) | fresh |
| ejkernel-types-mask | [ejkernel-types-mask](concepts/ejkernel-types-mask.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **3430** across 216 modules
- deep (concept pages): **325** (9.5%)
- catalog-only: **3105**
- represented total: **3430** (100.0%)
- classes represented: **174/174**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
