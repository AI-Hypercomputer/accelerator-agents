---
slug: graphcast
commit: 08cf73625c9d12bd9aaa038868bcb2fe488f2a22
scip_tool: scip-python
updated: 2026-07-03
---

# graphcast internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| graphcast-casting | [graphcast-casting](concepts/graphcast-casting.md) | fresh |
| graphcast-deep_typed_graph_net | [graphcast-deep_typed_graph_net](concepts/graphcast-deep_typed_graph_net.md) | fresh |
| graphcast-sparse_transformer | [graphcast-sparse_transformer](concepts/graphcast-sparse_transformer.md) | fresh |
| graphcast-typed_graph | [graphcast-typed_graph](concepts/graphcast-typed_graph.md) | fresh |
| graphcast | [graphcast](concepts/graphcast.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **781** across 37 modules
- deep (concept pages): **56** (7.2%)
- catalog-only: **725**
- represented total: **781** (100.0%)
- classes represented: **55/55**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
