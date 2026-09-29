---
slug: tpu-inference
commit: 8eb31eeee761a8056104acf10b312bae19e72165
scip_tool: scip-python
updated: 2026-07-03
---

# tpu-inference internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| root | [root](concepts/root.md) | fresh |
| tpu_inference-kernels-mla-v2-kernel | [tpu_inference-kernels-mla-v2-kernel](concepts/tpu_inference-kernels-mla-v2-kernel.md) | fresh |
| tpu_inference-layers-common-attention_metadata | [tpu_inference-layers-common-attention_metadata](concepts/tpu_inference-layers-common-attention_metadata.md) | fresh |
| tpu_inference-layers-jax-attention | [tpu_inference-layers-jax-attention](concepts/tpu_inference-layers-jax-attention.md) | fresh |
| tpu_inference-layers-jax-moe | [tpu_inference-layers-jax-moe](concepts/tpu_inference-layers-jax-moe.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **4786** across 208 modules
- deep (concept pages): **55** (1.1%)
- catalog-only: **4731**
- represented total: **4786** (100.0%)
- classes represented: **346/346**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
