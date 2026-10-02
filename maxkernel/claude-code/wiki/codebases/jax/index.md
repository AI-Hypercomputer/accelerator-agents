---
slug: jax
commit: 415e5b70078f509fee8a0de498fdc7c37b92f6c6
scip_tool: scip-python
updated: 2026-07-03
---

# jax internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| jax-_src-api | [jax-_src-api](concepts/jax-_src-api.md) | fresh |
| jax-_src-basearray | [jax-_src-basearray](concepts/jax-_src-basearray.md) | fresh |
| jax-_src-config | [jax-_src-config](concepts/jax-_src-config.md) | fresh |
| jax-_src-core | [jax-_src-core](concepts/jax-_src-core.md) | fresh |
| jax-_src-dtypes | [jax-_src-dtypes](concepts/jax-_src-dtypes.md) | fresh |
| jax-_src-lax | [jax-_src-lax](concepts/jax-_src-lax.md) | fresh |
| jax-_src-mesh | [jax-_src-mesh](concepts/jax-_src-mesh.md) | fresh |
| jax-_src-named_sharding | [jax-_src-named_sharding](concepts/jax-_src-named_sharding.md) | fresh |
| jax-_src-numpy-lax_numpy | [jax-_src-numpy-lax_numpy](concepts/jax-_src-numpy-lax_numpy.md) | fresh |
| jax-_src-pallas-core | [jax-_src-pallas-core](concepts/jax-_src-pallas-core.md) | fresh |
| jax-_src-pallas-fuser-block_spec | [jax-_src-pallas-fuser-block_spec](concepts/jax-_src-pallas-fuser-block_spec.md) | fresh |
| jax-_src-pallas-mosaic-lowering | [jax-_src-pallas-mosaic-lowering](concepts/jax-_src-pallas-mosaic-lowering.md) | fresh |
| jax-_src-pallas-mosaic_gpu-core | [jax-_src-pallas-mosaic_gpu-core](concepts/jax-_src-pallas-mosaic_gpu-core.md) | fresh |
| jax-_src-pallas-mosaic_gpu-lowering | [jax-_src-pallas-mosaic_gpu-lowering](concepts/jax-_src-pallas-mosaic_gpu-lowering.md) | fresh |
| jax-_src-pallas-mosaic_gpu-primitives | [jax-_src-pallas-mosaic_gpu-primitives](concepts/jax-_src-pallas-mosaic_gpu-primitives.md) | fresh |
| jax-_src-pallas-triton-lowering | [jax-_src-pallas-triton-lowering](concepts/jax-_src-pallas-triton-lowering.md) | fresh |
| jax-_src-state-types | [jax-_src-state-types](concepts/jax-_src-state-types.md) | fresh |
| jax-_src-tree_util | [jax-_src-tree_util](concepts/jax-_src-tree_util.md) | fresh |
| jax-_src-util | [jax-_src-util](concepts/jax-_src-util.md) | fresh |
| jax-experimental-mosaic-gpu-core | [jax-experimental-mosaic-gpu-core](concepts/jax-experimental-mosaic-gpu-core.md) | fresh |
| jax-experimental-mosaic-gpu-fragmented_array | [jax-experimental-mosaic-gpu-fragmented_array](concepts/jax-experimental-mosaic-gpu-fragmented_array.md) | fresh |
| jax-experimental-mosaic-gpu-layout_inference | [jax-experimental-mosaic-gpu-layout_inference](concepts/jax-experimental-mosaic-gpu-layout_inference.md) | fresh |
| jax-experimental-mosaic-gpu-utils | [jax-experimental-mosaic-gpu-utils](concepts/jax-experimental-mosaic-gpu-utils.md) | fresh |
| jax-numpy-__init__.pyi | [jax-numpy-__init__.pyi](concepts/jax-numpy-__init__.pyi.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **13031** across 211 modules
- deep (concept pages): **103** (0.8%)
- catalog-only: **12928**
- represented total: **13031** (100.0%)
- classes represented: **660/660**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
