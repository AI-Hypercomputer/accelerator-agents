---
slug: learning-machine
commit: 9ad94857b12995f2cdbadef56744bca35006c4c9
scip_tool: scip-python
updated: 2026-07-03
---

# learning-machine internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| basic_ops | [basic_ops](concepts/basic_ops.md) | fresh |
| custom_kernel_spmd | [custom_kernel_spmd](concepts/custom_kernel_spmd.md) | fresh |
| dlpack-collect_env | [dlpack-collect_env](concepts/dlpack-collect_env.md) | fresh |
| equinox_with_torch | [equinox_with_torch](concepts/equinox_with_torch.md) | fresh |
| jax_perf-splash_attention_debug | [jax_perf-splash_attention_debug](concepts/jax_perf-splash_attention_debug.md) | fresh |
| llama_ref-model | [llama_ref-model](concepts/llama_ref-model.md) | fresh |
| llama_ref-model_original | [llama_ref-model_original](concepts/llama_ref-model_original.md) | fresh |
| llama_ref-model_with_collectives | [llama_ref-model_with_collectives](concepts/llama_ref-model_with_collectives.md) | fresh |
| llama_ref-model_with_scan | [llama_ref-model_with_scan](concepts/llama_ref-model_with_scan.md) | fresh |
| spmd_sharding_jax-ffn_jax | [spmd_sharding_jax-ffn_jax](concepts/spmd_sharding_jax-ffn_jax.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **803** across 40 modules
- deep (concept pages): **201** (25.0%)
- catalog-only: **602**
- represented total: **803** (100.0%)
- classes represented: **46/46**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
