---
slug: sglang-jax
commit: ada34734adb62810abb9d065a2787ba2bddbcbc0
scip_tool: scip-python
updated: 2026-07-03
---

# sglang-jax internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| python-sgl_jax-srt-configs-model_config | [python-sgl_jax-srt-configs-model_config](concepts/python-sgl_jax-srt-configs-model_config.md) | fresh |
| python-sgl_jax-srt-disaggregation-jax_transfer-conn | [python-sgl_jax-srt-disaggregation-jax_transfer-conn](concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md) | fresh |
| python-sgl_jax-srt-kernels-fused_moe-v1-kernel | [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md) | fresh |
| python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare | [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md) | fresh |
| python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2 | [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md) | fresh |
| python-sgl_jax-srt-kernels-fused_moe-v2-kernel | [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md) | fresh |
| python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2 | [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md) | fresh |
| python-sgl_jax-srt-layers-logits_processor | [python-sgl_jax-srt-layers-logits_processor](concepts/python-sgl_jax-srt-layers-logits_processor.md) | fresh |
| python-sgl_jax-srt-managers-io_struct | [python-sgl_jax-srt-managers-io_struct](concepts/python-sgl_jax-srt-managers-io_struct.md) | fresh |
| python-sgl_jax-srt-managers-schedule_policy | [python-sgl_jax-srt-managers-schedule_policy](concepts/python-sgl_jax-srt-managers-schedule_policy.md) | fresh |
| python-sgl_jax-srt-managers-scheduler | [python-sgl_jax-srt-managers-scheduler](concepts/python-sgl_jax-srt-managers-scheduler.md) | fresh |
| python-sgl_jax-srt-mem_cache-allocator | [python-sgl_jax-srt-mem_cache-allocator](concepts/python-sgl_jax-srt-mem_cache-allocator.md) | fresh |
| python-sgl_jax-srt-mem_cache-base_prefix_cache | [python-sgl_jax-srt-mem_cache-base_prefix_cache](concepts/python-sgl_jax-srt-mem_cache-base_prefix_cache.md) | fresh |
| python-sgl_jax-srt-mem_cache-memory_pool | [python-sgl_jax-srt-mem_cache-memory_pool](concepts/python-sgl_jax-srt-mem_cache-memory_pool.md) | fresh |
| python-sgl_jax-srt-mem_cache-radix_cache | [python-sgl_jax-srt-mem_cache-radix_cache](concepts/python-sgl_jax-srt-mem_cache-radix_cache.md) | fresh |
| python-sgl_jax-srt-mem_cache-swa_radix_cache | [python-sgl_jax-srt-mem_cache-swa_radix_cache](concepts/python-sgl_jax-srt-mem_cache-swa_radix_cache.md) | fresh |
| python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component | [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md) | fresh |
| python-sgl_jax-srt-mem_cache-unified_radix_cache | [python-sgl_jax-srt-mem_cache-unified_radix_cache](concepts/python-sgl_jax-srt-mem_cache-unified_radix_cache.md) | fresh |
| python-sgl_jax-srt-model_executor-forward_batch_info | [python-sgl_jax-srt-model_executor-forward_batch_info](concepts/python-sgl_jax-srt-model_executor-forward_batch_info.md) | fresh |
| python-sgl_jax-srt-model_executor-model_runner | [python-sgl_jax-srt-model_executor-model_runner](concepts/python-sgl_jax-srt-model_executor-model_runner.md) | fresh |
| python-sgl_jax-srt-server_args | [python-sgl_jax-srt-server_args](concepts/python-sgl_jax-srt-server_args.md) | fresh |
| python-sgl_jax-srt-speculative-eagle_util | [python-sgl_jax-srt-speculative-eagle_util](concepts/python-sgl_jax-srt-speculative-eagle_util.md) | fresh |
| python-sgl_jax-srt-utils-weight_utils | [python-sgl_jax-srt-utils-weight_utils](concepts/python-sgl_jax-srt-utils-weight_utils.md) | fresh |
| root | [root](concepts/root.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **6498** across 185 modules
- deep (concept pages): **307** (4.7%)
- catalog-only: **6191**
- represented total: **6498** (100.0%)
- classes represented: **384/384**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
