---
slug: aqt
commit: 9d1667ebde057f474cf8d2c80d4e52c5670d6c8a
scip_tool: scip-python
updated: 2026-07-03
---

# aqt internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| aqt-jax-aqt_dot_general | [aqt-jax-aqt_dot_general](concepts/aqt-jax-aqt_dot_general.md) | fresh |
| aqt-jax-v2-aqt_quantizer | [aqt-jax-v2-aqt_quantizer](concepts/aqt-jax-v2-aqt_quantizer.md) | fresh |
| aqt-jax-v2-aqt_tensor | [aqt-jax-v2-aqt_tensor](concepts/aqt-jax-v2-aqt_tensor.md) | fresh |
| aqt-jax-v2-numerics-fp_numerics | [aqt-jax-v2-numerics-fp_numerics](concepts/aqt-jax-v2-numerics-fp_numerics.md) | fresh |
| aqt-jax-v2-utils | [aqt-jax-v2-utils](concepts/aqt-jax-v2-utils.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **760** across 48 modules
- deep (concept pages): **87** (11.4%)
- catalog-only: **673**
- represented total: **760** (100.0%)
- classes represented: **77/77**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
