---
slug: xprof-mcp
commit: de25d4a18ad04e416a989c819b40845b10f3d9c5
scip_tool: scip-python
updated: 2026-07-03
---

# xprof-mcp internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| internal-xprof_client | [internal-xprof_client](concepts/internal-xprof_client.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **75** across 10 modules
- deep (concept pages): **20** (26.7%)
- catalog-only: **55**
- represented total: **75** (100.0%)
- classes represented: **1/1**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
