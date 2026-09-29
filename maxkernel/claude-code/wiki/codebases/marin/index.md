---
slug: marin
commit: 240bf20a8de294e568dba5d14c4f02034e249b21
scip_tool: scip-python
updated: 2026-07-03
---

# marin internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| lib-levanter-src-levanter-checkpoint | [lib-levanter-src-levanter-checkpoint](concepts/lib-levanter-src-levanter-checkpoint.md) | fresh |
| lib-levanter-src-levanter-config | [lib-levanter-src-levanter-config](concepts/lib-levanter-src-levanter-config.md) | fresh |
| lib-levanter-src-levanter-inference-engine | [lib-levanter-src-levanter-inference-engine](concepts/lib-levanter-src-levanter-inference-engine.md) | fresh |
| lib-levanter-src-levanter-inference-jit_scheduler | [lib-levanter-src-levanter-inference-jit_scheduler](concepts/lib-levanter-src-levanter-inference-jit_scheduler.md) | fresh |
| lib-levanter-src-levanter-kernels-deepep-transport_ffi | [lib-levanter-src-levanter-kernels-deepep-transport_ffi](concepts/lib-levanter-src-levanter-kernels-deepep-transport_ffi.md) | fresh |
| lib-levanter-src-levanter-layers-attention_mask | [lib-levanter-src-levanter-layers-attention_mask](concepts/lib-levanter-src-levanter-layers-attention_mask.md) | fresh |
| lib-levanter-src-levanter-layers-rotary | [lib-levanter-src-levanter-layers-rotary](concepts/lib-levanter-src-levanter-layers-rotary.md) | fresh |
| lib-levanter-src-levanter-models-llama | [lib-levanter-src-levanter-models-llama](concepts/lib-levanter-src-levanter-models-llama.md) | fresh |
| lib-levanter-src-levanter-models-lm_model | [lib-levanter-src-levanter-models-lm_model](concepts/lib-levanter-src-levanter-models-lm_model.md) | fresh |
| lib-levanter-src-levanter-models-olmo | [lib-levanter-src-levanter-models-olmo](concepts/lib-levanter-src-levanter-models-olmo.md) | fresh |
| lib-levanter-src-levanter-trainer | [lib-levanter-src-levanter-trainer](concepts/lib-levanter-src-levanter-trainer.md) | fresh |
| root | [root](concepts/root.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **5954** across 177 modules
- deep (concept pages): **158** (2.7%)
- catalog-only: **5796**
- represented total: **5954** (100.0%)
- classes represented: **452/452**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
