---
title: 'Module: tpu_inference/offload/utils.py'
type: catalog
provenance: extracted
module: tpu_inference/offload/utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.offload.utils`/
symbols:
  update_kv_caches_one: update_kv_caches_one().
  get_kv_connector_cache_layout: get_kv_connector_cache_layout().
  logger: logger.
  jitted_insert_kv_cache_slices: jitted_insert_kv_cache_slices().
  stack_kv_cache_cross_layers: stack_kv_cache_cross_layers().
  pure_jax_stack_kv_cache_cross_layers: pure_jax_stack_kv_cache_cross_layers().
  pure_jax_update_kv_caches_one: pure_jax_update_kv_caches_one().
  jitted_insert_kv_cache_slices._update_layer: jitted_insert_kv_cache_slices()._update_layer().
  stack_kv_cache_cross_layers._gather_blocks: stack_kv_cache_cross_layers()._gather_blocks().
  pre_update_kv_caches: pre_update_kv_caches().
  update_kv_caches: update_kv_caches().
  pure_jax_stack_kv_cache_cross_layers._gather_blocks: pure_jax_stack_kv_cache_cross_layers()._gather_blocks().
  pure_jax_update_kv_caches_one._update_layer: pure_jax_update_kv_caches_one()._update_layer().
  ReqId: ReqId.
  CpuChunkId: CpuChunkId.
  NONE_HASH: NONE_HASH.
---
# Module: [`tpu_inference/offload/utils.py`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py)

## Functions
- `_gather_blocks(layer_kv_cache)` — [`L104`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L104)
- `_gather_blocks(layer_kv_cache)` — [`L299`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L299)
- `_update_layer(cache, slices)` — [`L77`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L77) — The function to apply to each layer's cache and slices.
- `_update_layer(cache_layer, slices)` — [`L369`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L369)
- `get_kv_connector_cache_layout()` — [`L38`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L38) — Retrieve the required kv cache layout for the configured kv connector
- `jitted_insert_kv_cache_slices(block_size, kv_caches: List[jax.Array], kv_cache_slices: List[List[jax.Array]], block_numbers: jax.Array)` — [`L65`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L65) — JIT-compiled function to insert KV cache slices into the physical
- `pre_update_kv_caches(block_indices: List[int], mesh: Mesh, replicated_sharding: PartitionSpec | None = None)` — [`L201`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L201)
- `pure_jax_stack_kv_cache_cross_layers(kv_caches: List[jax.Array], block_ids: jax.Array, num_blocks: int)` — [`L277`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L277) — Gathers KV cache blocks across all layers for offloading, using pure JAX operations.
- `pure_jax_update_kv_caches_one(kv_caches: List[jax.Array], stacked_blocks: List[jax.Array], block_indices: List[int], mesh: Mesh, cached_kv_sharding_spec: PartitionSpec | None = None, indices_sharding: PartitionSpec | None = None)` — [`L318`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L318) — Updates the physical KV cache by inserting stacked blocks into specified
- `stack_kv_cache_cross_layers(kv_caches: List[jax.Array], block_ids: jax.Array, num_blocks: int)` — [`L97`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L97) — This uses jax.tree.map to apply the operation across all layers.
- `update_kv_caches(kv_caches: List[jax.Array], stacked_blocks: List[jax.Array], src_offsets: jax.Array, dest_offsets: jax.Array, chunk_sizes: jax.Array, num_chunks: jax.Array, mesh, src_sharding_spec, dest_sharding_spec, replicated_sharding_spec)` — [`L230`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L230) — Updates KV caches by unstacking gathered blocks and copying slices.
- `update_kv_caches_one(kv_caches: List[jax.Array], stacked_blocks: List[jax.Array], block_indices: List[int], mesh: Mesh, cached_kv_sharding_spec: PartitionSpec | None = None, replicated_sharding: PartitionSpec | None = None)` — [`L162`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L162) — Update KV caches using cached sharding spec to avoid recompilation.

## Module values
- `CpuChunkId` — [`L30`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L30)
- `NONE_HASH` — [`L33`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L33)
- `ReqId` — [`L28`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L28)
- `logger` — [`L35`](../../../../../../raw/code/tpu-inference/tpu_inference/offload/utils.py#L35)

