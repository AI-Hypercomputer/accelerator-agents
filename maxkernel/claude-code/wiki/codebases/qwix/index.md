---
slug: qwix
commit: e621789c2ec3b1d624d8a4f71571207b6947937d
scip_tool: scip-python
updated: 2026-07-03
---

# qwix internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| qwix-_src-averaging | [qwix-_src-averaging](concepts/qwix-_src-averaging.md) | fresh |
| qwix-_src-core-conv_general_qt | [qwix-_src-core-conv_general_qt](concepts/qwix-_src-core-conv_general_qt.md) | fresh |
| qwix-_src-core-dot_general | [qwix-_src-core-dot_general](concepts/qwix-_src-core-dot_general.md) | fresh |
| qwix-_src-core-dot_general_qt | [qwix-_src-core-dot_general_qt](concepts/qwix-_src-core-dot_general_qt.md) | fresh |
| qwix-_src-core-einsum_info | [qwix-_src-core-einsum_info](concepts/qwix-_src-core-einsum_info.md) | fresh |
| qwix-_src-core-numerics | [qwix-_src-core-numerics](concepts/qwix-_src-core-numerics.md) | fresh |
| qwix-_src-core-qarray | [qwix-_src-core-qarray](concepts/qwix-_src-core-qarray.md) | fresh |
| qwix-_src-core-sparsity | [qwix-_src-core-sparsity](concepts/qwix-_src-core-sparsity.md) | fresh |
| qwix-_src-interception | [qwix-_src-interception](concepts/qwix-_src-interception.md) | fresh |
| qwix-_src-model | [qwix-_src-model](concepts/qwix-_src-model.md) | fresh |
| qwix-_src-providers-lora | [qwix-_src-providers-lora](concepts/qwix-_src-providers-lora.md) | fresh |
| qwix-_src-providers-odml | [qwix-_src-providers-odml](concepts/qwix-_src-providers-odml.md) | fresh |
| qwix-_src-providers-odml_ops | [qwix-_src-providers-odml_ops](concepts/qwix-_src-providers-odml_ops.md) | fresh |
| qwix-_src-providers-ptq | [qwix-_src-providers-ptq](concepts/qwix-_src-providers-ptq.md) | fresh |
| qwix-_src-providers-qt | [qwix-_src-providers-qt](concepts/qwix-_src-providers-qt.md) | fresh |
| qwix-_src-qconfig | [qwix-_src-qconfig](concepts/qwix-_src-qconfig.md) | fresh |
| qwix-_src-utils-checkpoint_util | [qwix-_src-utils-checkpoint_util](concepts/qwix-_src-utils-checkpoint_util.md) | fresh |
| qwix-_src-utils-flax_util | [qwix-_src-utils-flax_util](concepts/qwix-_src-utils-flax_util.md) | fresh |
| qwix-contrib-awq | [qwix-contrib-awq](concepts/qwix-contrib-awq.md) | fresh |
| qwix-contrib-calibration | [qwix-contrib-calibration](concepts/qwix-contrib-calibration.md) | fresh |
| qwix-contrib-gptq | [qwix-contrib-gptq](concepts/qwix-contrib-gptq.md) | fresh |
| qwix-contrib-padded_ptq | [qwix-contrib-padded_ptq](concepts/qwix-contrib-padded_ptq.md) | fresh |
| qwix-contrib-qep | [qwix-contrib-qep](concepts/qwix-contrib-qep.md) | fresh |
| qwix-contrib-smooth_quant | [qwix-contrib-smooth_quant](concepts/qwix-contrib-smooth_quant.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **1561** across 93 modules
- deep (concept pages): **236** (15.1%)
- catalog-only: **1325**
- represented total: **1561** (100.0%)
- classes represented: **213/213**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
