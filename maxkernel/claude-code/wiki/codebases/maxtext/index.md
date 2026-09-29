---
slug: maxtext
commit: fcb7ebeba9ecfc67d79e471f50c16c9d89b3263d
scip_tool: scip-python
updated: 2026-07-03
---

# maxtext internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| maxtext-configs-pyconfig_deprecated | [maxtext-configs-pyconfig_deprecated](concepts/maxtext-configs-pyconfig_deprecated.md) | fresh |
| maxtext-configs-types | [maxtext-configs-types](concepts/maxtext-configs-types.md) | fresh |
| maxtext-input_pipeline-input_pipeline_utils | [maxtext-input_pipeline-input_pipeline_utils](concepts/maxtext-input_pipeline-input_pipeline_utils.md) | fresh |
| maxtext-input_pipeline-olmo_data | [maxtext-input_pipeline-olmo_data](concepts/maxtext-input_pipeline-olmo_data.md) | fresh |
| maxtext-input_pipeline-packing-prefill_packing | [maxtext-input_pipeline-packing-prefill_packing](concepts/maxtext-input_pipeline-packing-prefill_packing.md) | fresh |
| maxtext-kernels-attention-splash_attention_kernel | [maxtext-kernels-attention-splash_attention_kernel](concepts/maxtext-kernels-attention-splash_attention_kernel.md) | fresh |
| maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel | [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md) | fresh |
| maxtext-layers-attention_compressed | [maxtext-layers-attention_compressed](concepts/maxtext-layers-attention_compressed.md) | fresh |
| maxtext-layers-attention_mla | [maxtext-layers-attention_mla](concepts/maxtext-layers-attention_mla.md) | fresh |
| maxtext-layers-attention_op | [maxtext-layers-attention_op](concepts/maxtext-layers-attention_op.md) | fresh |
| maxtext-layers-attentions | [maxtext-layers-attentions](concepts/maxtext-layers-attentions.md) | fresh |
| maxtext-layers-decoders | [maxtext-layers-decoders](concepts/maxtext-layers-decoders.md) | fresh |
| maxtext-layers-embeddings | [maxtext-layers-embeddings](concepts/maxtext-layers-embeddings.md) | fresh |
| maxtext-layers-linears | [maxtext-layers-linears](concepts/maxtext-layers-linears.md) | fresh |
| maxtext-layers-moe | [maxtext-layers-moe](concepts/maxtext-layers-moe.md) | fresh |
| maxtext-layers-nnx_decoders | [maxtext-layers-nnx_decoders](concepts/maxtext-layers-nnx_decoders.md) | fresh |
| maxtext-layers-pipeline | [maxtext-layers-pipeline](concepts/maxtext-layers-pipeline.md) | fresh |
| maxtext-layers-pipeline_deprecated | [maxtext-layers-pipeline_deprecated](concepts/maxtext-layers-pipeline_deprecated.md) | fresh |
| maxtext-models-deepseek | [maxtext-models-deepseek](concepts/maxtext-models-deepseek.md) | fresh |
| maxtext-models-deepseek_batchsplit | [maxtext-models-deepseek_batchsplit](concepts/maxtext-models-deepseek_batchsplit.md) | fresh |
| maxtext-models-gemma3 | [maxtext-models-gemma3](concepts/maxtext-models-gemma3.md) | fresh |
| maxtext-models-gpt3 | [maxtext-models-gpt3](concepts/maxtext-models-gpt3.md) | fresh |
| maxtext-models-llama4 | [maxtext-models-llama4](concepts/maxtext-models-llama4.md) | fresh |
| maxtext-models-qwen3 | [maxtext-models-qwen3](concepts/maxtext-models-qwen3.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **5425** across 123 modules
- deep (concept pages): **963** (17.8%)
- catalog-only: **4462**
- represented total: **5425** (100.0%)
- classes represented: **376/376**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
