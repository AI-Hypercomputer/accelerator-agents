---
slug: pallas-forge
commit: 090510b7b36796a31f7bacee04c31d7e1909cf76
scip_tool: scip-python
updated: 2026-07-03
---

# pallas-forge internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| pallas_forge-tune-config | [pallas_forge-tune-config](concepts/pallas_forge-tune-config.md) | fresh |
| pallas_forge-tune-report | [pallas_forge-tune-report](concepts/pallas_forge-tune-report.md) | fresh |
| pallas_forge-tune-runner | [pallas_forge-tune-runner](concepts/pallas_forge-tune-runner.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **197** across 29 modules
- deep (concept pages): **51** (25.9%)
- catalog-only: **146**
- represented total: **197** (100.0%)
- classes represented: **19/19**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
