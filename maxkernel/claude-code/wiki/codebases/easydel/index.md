---
slug: easydel
commit: 090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246
scip_tool: scip-python
updated: 2026-07-04
---

# easydel internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| easydel-caching-_abstracts | [easydel-caching-_abstracts](concepts/easydel-caching-_abstracts.md) | fresh |
| easydel-caching-hybrid-cache | [easydel-caching-hybrid-cache](concepts/easydel-caching-hybrid-cache.md) | fresh |
| easydel-caching-ragged_page-cache | [easydel-caching-ragged_page-cache](concepts/easydel-caching-ragged_page-cache.md) | fresh |
| easydel-caching-transformer-cache | [easydel-caching-transformer-cache](concepts/easydel-caching-transformer-cache.md) | fresh |
| easydel-inference-esurge-esurge_engine | [easydel-inference-esurge-esurge_engine](concepts/easydel-inference-esurge-esurge_engine.md) | fresh |
| easydel-inference-esurge-runners-sequence_buffer | [easydel-inference-esurge-runners-sequence_buffer](concepts/easydel-inference-esurge-runners-sequence_buffer.md) | fresh |
| easydel-inference-openai_api_modules | [easydel-inference-openai_api_modules](concepts/easydel-inference-openai_api_modules.md) | fresh |
| easydel-infra-base_config | [easydel-infra-base_config](concepts/easydel-infra-base_config.md) | fresh |
| easydel-infra-base_module | [easydel-infra-base_module](concepts/easydel-infra-base_module.md) | fresh |
| easydel-infra-base_state | [easydel-infra-base_state](concepts/easydel-infra-base_state.md) | fresh |
| easydel-infra-factory | [easydel-infra-factory](concepts/easydel-infra-factory.md) | fresh |
| easydel-infra-loss_utils | [easydel-infra-loss_utils](concepts/easydel-infra-loss_utils.md) | fresh |
| easydel-infra-modeling_outputs | [easydel-infra-modeling_outputs](concepts/easydel-infra-modeling_outputs.md) | fresh |
| easydel-infra-utils | [easydel-infra-utils](concepts/easydel-infra-utils.md) | fresh |
| easydel-layers-attention-_unified | [easydel-layers-attention-_unified](concepts/easydel-layers-attention-_unified.md) | fresh |
| easydel-layers-linears-_linear | [easydel-layers-linears-_linear](concepts/easydel-layers-linears-_linear.md) | fresh |
| easydel-layers-norms-_norms | [easydel-layers-norms-_norms](concepts/easydel-layers-norms-_norms.md) | fresh |
| easydel-modules-gemma4-modeling_gemma4 | [easydel-modules-gemma4-modeling_gemma4](concepts/easydel-modules-gemma4-modeling_gemma4.md) | fresh |
| easydel-modules-qwen3_omni_moe-modeling_qwen3_omni_moe | [easydel-modules-qwen3_omni_moe-modeling_qwen3_omni_moe](concepts/easydel-modules-qwen3_omni_moe-modeling_qwen3_omni_moe.md) | fresh |
| easydel-modules-qwen3_omni_moe-qwen3_omni_moe_configuration | [easydel-modules-qwen3_omni_moe-qwen3_omni_moe_configuration](concepts/easydel-modules-qwen3_omni_moe-qwen3_omni_moe_configuration.md) | fresh |
| easydel-operations-requirements-types | [easydel-operations-requirements-types](concepts/easydel-operations-requirements-types.md) | fresh |
| easydel-trainers-base_trainer | [easydel-trainers-base_trainer](concepts/easydel-trainers-base_trainer.md) | fresh |
| easydel-trainers-training_configurations | [easydel-trainers-training_configurations](concepts/easydel-trainers-training_configurations.md) | fresh |
| easydel-workers-esurge-auth-auth_models | [easydel-workers-esurge-auth-auth_models](concepts/easydel-workers-esurge-auth-auth_models.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **21412** across 642 modules
- deep (concept pages): **259** (1.2%)
- catalog-only: **21153**
- represented total: **21412** (100.0%)
- classes represented: **1582/1582**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
