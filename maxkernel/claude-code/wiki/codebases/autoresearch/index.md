---
slug: autoresearch
commit: 228791fb499afffb54b46200aca536f79142f117
scip_tool: scip-python
updated: 2026-07-04
---

# autoresearch internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| prepare | [prepare](concepts/prepare.md) | fresh |
| train | [train](concepts/train.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **177** across 2 modules
- deep (concept pages): **75** (42.4%)
- catalog-only: **102**
- represented total: **177** (100.0%)
- classes represented: **7/7**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
