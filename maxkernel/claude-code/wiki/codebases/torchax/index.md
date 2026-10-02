---
slug: torchax
commit: ac59e3d2becf079b336a8caa831746bb0326e6d4
scip_tool: scip-python
updated: 2026-07-03
---

# torchax internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| docs-docs-tutorials-distributed_array | [docs-docs-tutorials-distributed_array](concepts/docs-docs-tutorials-distributed_array.md) | fresh |
| docs-docs-tutorials-trainingyt | [docs-docs-tutorials-trainingyt](concepts/docs-docs-tutorials-trainingyt.md) | fresh |
| torchax-export | [torchax-export](concepts/torchax-export.md) | fresh |
| torchax-interop | [torchax-interop](concepts/torchax-interop.md) | fresh |
| torchax-ops-jaten | [torchax-ops-jaten](concepts/torchax-ops-jaten.md) | fresh |
| torchax-ops-jtorch | [torchax-ops-jtorch](concepts/torchax-ops-jtorch.md) | fresh |
| torchax-ops-mappings | [torchax-ops-mappings](concepts/torchax-ops-mappings.md) | fresh |
| torchax-ops-op_base | [torchax-ops-op_base](concepts/torchax-ops-op_base.md) | fresh |
| torchax-ops-ops_registry | [torchax-ops-ops_registry](concepts/torchax-ops-ops_registry.md) | fresh |
| torchax-tensor | [torchax-tensor](concepts/torchax-tensor.md) | fresh |
| torchax-view | [torchax-view](concepts/torchax-view.md) | fresh |
| torchax | [torchax](concepts/torchax.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **2150** across 70 modules
- deep (concept pages): **159** (7.4%)
- catalog-only: **1991**
- represented total: **2150** (100.0%)
- classes represented: **114/114**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
