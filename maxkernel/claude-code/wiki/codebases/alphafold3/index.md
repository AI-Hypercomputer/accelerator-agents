---
slug: alphafold3
commit: b1f78022340f257ff9bbffde649f8b779208ec29
scip_tool: scip-python
updated: 2026-07-03
---

# alphafold3 internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| alphafold3-jax-geometry-rigid_matrix_vector | [alphafold3-jax-geometry-rigid_matrix_vector](concepts/alphafold3-jax-geometry-rigid_matrix_vector.md) | fresh |
| alphafold3-jax-geometry-rotation_matrix | [alphafold3-jax-geometry-rotation_matrix](concepts/alphafold3-jax-geometry-rotation_matrix.md) | fresh |
| alphafold3-jax-geometry-vector | [alphafold3-jax-geometry-vector](concepts/alphafold3-jax-geometry-vector.md) | fresh |
| alphafold3-model-atom_layout | [alphafold3-model-atom_layout](concepts/alphafold3-model-atom_layout.md) | fresh |
| alphafold3-model-components-haiku_modules | [alphafold3-model-components-haiku_modules](concepts/alphafold3-model-components-haiku_modules.md) | fresh |
| alphafold3-model-components-mapping | [alphafold3-model-components-mapping](concepts/alphafold3-model-components-mapping.md) | fresh |
| alphafold3-model-confidence_types | [alphafold3-model-confidence_types](concepts/alphafold3-model-confidence_types.md) | fresh |
| alphafold3-model-feat_batch | [alphafold3-model-feat_batch](concepts/alphafold3-model-feat_batch.md) | fresh |
| alphafold3-model-features | [alphafold3-model-features](concepts/alphafold3-model-features.md) | fresh |
| alphafold3-model-model_config | [alphafold3-model-model_config](concepts/alphafold3-model-model_config.md) | fresh |
| alphafold3-model-network-atom_cross_attention | [alphafold3-model-network-atom_cross_attention](concepts/alphafold3-model-network-atom_cross_attention.md) | fresh |
| alphafold3-model-network-diffusion_head | [alphafold3-model-network-diffusion_head](concepts/alphafold3-model-network-diffusion_head.md) | fresh |
| alphafold3-model-network-diffusion_transformer | [alphafold3-model-network-diffusion_transformer](concepts/alphafold3-model-network-diffusion_transformer.md) | fresh |
| alphafold3-model-network-evoformer | [alphafold3-model-network-evoformer](concepts/alphafold3-model-network-evoformer.md) | fresh |
| alphafold3-model-network-modules | [alphafold3-model-network-modules](concepts/alphafold3-model-network-modules.md) | fresh |
| alphafold3-model-network-template_modules | [alphafold3-model-network-template_modules](concepts/alphafold3-model-network-template_modules.md) | fresh |
| alphafold3-model | [alphafold3-model](concepts/alphafold3-model.md) | fresh |
| alphafold3-structure-bioassemblies | [alphafold3-structure-bioassemblies](concepts/alphafold3-structure-bioassemblies.md) | fresh |
| alphafold3-structure-bonds | [alphafold3-structure-bonds](concepts/alphafold3-structure-bonds.md) | fresh |
| alphafold3-structure-chemical_components | [alphafold3-structure-chemical_components](concepts/alphafold3-structure-chemical_components.md) | fresh |
| alphafold3-structure-mmcif | [alphafold3-structure-mmcif](concepts/alphafold3-structure-mmcif.md) | fresh |
| alphafold3-structure-parsing | [alphafold3-structure-parsing](concepts/alphafold3-structure-parsing.md) | fresh |
| alphafold3-structure-structure_tables | [alphafold3-structure-structure_tables](concepts/alphafold3-structure-structure_tables.md) | fresh |
| alphafold3-structure | [alphafold3-structure](concepts/alphafold3-structure.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **1236** across 47 modules
- deep (concept pages): **363** (29.4%)
- catalog-only: **873**
- represented total: **1236** (100.0%)
- classes represented: **110/110**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
