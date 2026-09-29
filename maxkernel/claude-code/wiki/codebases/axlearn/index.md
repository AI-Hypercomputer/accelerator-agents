---
slug: axlearn
commit: c8624c837c55f39d8e8fc4e513d54b7e19d108f6
scip_tool: scip-python
updated: 2026-07-03
---

# axlearn internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| axlearn-common-attention | [axlearn-common-attention](concepts/axlearn-common-attention.md) | fresh |
| axlearn-common-attention_bias | [axlearn-common-attention_bias](concepts/axlearn-common-attention_bias.md) | fresh |
| axlearn-common-flash_attention-common | [axlearn-common-flash_attention-common](concepts/axlearn-common-flash_attention-common.md) | fresh |
| axlearn-common-flash_attention-layer | [axlearn-common-flash_attention-layer](concepts/axlearn-common-flash_attention-layer.md) | fresh |
| axlearn-common-kv_cache-base_kv_cache | [axlearn-common-kv_cache-base_kv_cache](concepts/axlearn-common-kv_cache-base_kv_cache.md) | fresh |
| axlearn-common-trainer | [axlearn-common-trainer](concepts/axlearn-common-trainer.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **4625** across 119 modules
- deep (concept pages): **32** (0.7%)
- catalog-only: **4593**
- represented total: **4625** (100.0%)
- classes represented: **686/686**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
