---
slug: simply
commit: b1064e6f212de20eb5fec3ffcb9c6b6afa0d01a7
scip_tool: scip-python
updated: 2026-07-03
---

# simply internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| simply-agent-memory | [simply-agent-memory](concepts/simply-agent-memory.md) | fresh |
| simply-agent-tools | [simply-agent-tools](concepts/simply-agent-tools.md) | fresh |
| simply-agent-tui | [simply-agent-tui](concepts/simply-agent-tui.md) | fresh |
| simply-config_lib | [simply-config_lib](concepts/simply-config_lib.md) | fresh |
| simply-data_lib | [simply-data_lib](concepts/simply-data_lib.md) | fresh |
| simply-kernels-ragged_paged_attention | [simply-kernels-ragged_paged_attention](concepts/simply-kernels-ragged_paged_attention.md) | fresh |
| simply-model_lib | [simply-model_lib](concepts/simply-model_lib.md) | fresh |
| simply-rl_lib | [simply-rl_lib](concepts/simply-rl_lib.md) | fresh |
| simply-serving-page_batcher | [simply-serving-page_batcher](concepts/simply-serving-page_batcher.md) | fresh |
| simply-serving-vanilla_server | [simply-serving-vanilla_server](concepts/simply-serving-vanilla_server.md) | fresh |
| simply-utils-checkpoint_lib | [simply-utils-checkpoint_lib](concepts/simply-utils-checkpoint_lib.md) | fresh |
| simply-utils-common | [simply-utils-common](concepts/simply-utils-common.md) | fresh |
| simply-utils-experiment_helper | [simply-utils-experiment_helper](concepts/simply-utils-experiment_helper.md) | fresh |
| simply-utils-lm_format | [simply-utils-lm_format](concepts/simply-utils-lm_format.md) | fresh |
| simply-utils-module | [simply-utils-module](concepts/simply-utils-module.md) | fresh |
| simply-utils-moe_lib | [simply-utils-moe_lib](concepts/simply-utils-moe_lib.md) | fresh |
| simply-utils-optimizers | [simply-utils-optimizers](concepts/simply-utils-optimizers.md) | fresh |
| simply-utils-pytree | [simply-utils-pytree](concepts/simply-utils-pytree.md) | fresh |
| simply-utils-ragged_paged_attention | [simply-utils-ragged_paged_attention](concepts/simply-utils-ragged_paged_attention.md) | fresh |
| simply-utils-registry | [simply-utils-registry](concepts/simply-utils-registry.md) | fresh |
| simply-utils-replay_buffers | [simply-utils-replay_buffers](concepts/simply-utils-replay_buffers.md) | fresh |
| simply-utils-sampling_lib | [simply-utils-sampling_lib](concepts/simply-utils-sampling_lib.md) | fresh |
| simply-utils-sharding | [simply-utils-sharding](concepts/simply-utils-sharding.md) | fresh |
| simply-utils-tokenization | [simply-utils-tokenization](concepts/simply-utils-tokenization.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **3067** across 85 modules
- deep (concept pages): **307** (10.0%)
- catalog-only: **2760**
- represented total: **3067** (100.0%)
- classes represented: **305/305**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
