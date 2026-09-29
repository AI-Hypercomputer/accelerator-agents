---
slug: recurrentgemma
commit: 2efa84dac0e68e63547a27a18fa943c98f1c312e
scip_tool: scip-python
updated: 2026-07-03
---

# recurrentgemma internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| recurrentgemma-common | [recurrentgemma-common](concepts/recurrentgemma-common.md) | fresh |
| recurrentgemma-jax-array_typing | [recurrentgemma-jax-array_typing](concepts/recurrentgemma-jax-array_typing.md) | fresh |
| recurrentgemma-jax-complex_lib | [recurrentgemma-jax-complex_lib](concepts/recurrentgemma-jax-complex_lib.md) | fresh |
| recurrentgemma-jax-griffin | [recurrentgemma-jax-griffin](concepts/recurrentgemma-jax-griffin.md) | fresh |
| recurrentgemma-jax-layers | [recurrentgemma-jax-layers](concepts/recurrentgemma-jax-layers.md) | fresh |
| recurrentgemma-jax-modules | [recurrentgemma-jax-modules](concepts/recurrentgemma-jax-modules.md) | fresh |
| recurrentgemma-jax-pallas | [recurrentgemma-jax-pallas](concepts/recurrentgemma-jax-pallas.md) | fresh |
| recurrentgemma-jax-sampler | [recurrentgemma-jax-sampler](concepts/recurrentgemma-jax-sampler.md) | fresh |
| recurrentgemma-torch-array_typing | [recurrentgemma-torch-array_typing](concepts/recurrentgemma-torch-array_typing.md) | fresh |
| recurrentgemma-torch-layers | [recurrentgemma-torch-layers](concepts/recurrentgemma-torch-layers.md) | fresh |
| recurrentgemma-torch-modules | [recurrentgemma-torch-modules](concepts/recurrentgemma-torch-modules.md) | fresh |
| recurrentgemma-torch-sampler | [recurrentgemma-torch-sampler](concepts/recurrentgemma-torch-sampler.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **779** across 34 modules
- deep (concept pages): **262** (33.6%)
- catalog-only: **517**
- represented total: **779** (100.0%)
- classes represented: **62/62**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
