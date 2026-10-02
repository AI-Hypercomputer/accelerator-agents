---
title: 'Module: tpu_inference/runner/kv_cache_manager.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/kv_cache_manager.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.kv_cache_manager`/
symbols:
  KVCacheManager.runner: KVCacheManager#runner.
  KVCacheManager.initialize_kv_cache: KVCacheManager#initialize_kv_cache().
  KVCacheManager.get_kv_cache_spec: KVCacheManager#get_kv_cache_spec().
  KVCacheManager.insert_request_with_kv_cache: KVCacheManager#insert_request_with_kv_cache().
  KVCacheManager.maybe_reinitialize_input_batch: KVCacheManager#maybe_reinitialize_input_batch().
  KVCacheManager.update_mamba_page_size_padded: KVCacheManager#update_mamba_page_size_padded().
  KVCacheManager._create_attention_spec: KVCacheManager#_create_attention_spec().
  KVCacheManager._maybe_set_compact_mamba_num_blocks_override: KVCacheManager#_maybe_set_compact_mamba_num_blocks_override().
  logger: logger.
  KVCacheManager.delete_kv_cache: KVCacheManager#delete_kv_cache().
  KVCacheManager.reinitialize_kv_cache: KVCacheManager#reinitialize_kv_cache().
  KVCacheManager.transfer_kv_cache: KVCacheManager#transfer_kv_cache().
  KVCacheManager.use_mla: KVCacheManager#use_mla.
  KVCacheManager.get_kv_cache_for_block_ids: KVCacheManager#get_kv_cache_for_block_ids().
  KVCacheManager._jitted_insert_kv_cache: KVCacheManager#_jitted_insert_kv_cache().
  KVCacheManager: KVCacheManager#
  KVCacheManager.actual_mamba_num_blocks: KVCacheManager#actual_mamba_num_blocks.
  KVCacheManager._jitted_insert_continuous_kv_cache_from_slice: KVCacheManager#_jitted_insert_continuous_kv_cache_from_slice().
  KVCacheManager.shared_kv_cache_layers: KVCacheManager#shared_kv_cache_layers.
  KVCacheManager._hybrid_uniform_page_size_bytes: KVCacheManager#_hybrid_uniform_page_size_bytes.
  KVCacheManager._jitted_gather_kv_cache: KVCacheManager#_jitted_gather_kv_cache().
  KVCacheManager._jitted_gather_continuous_kv_cache: KVCacheManager#_jitted_gather_continuous_kv_cache().
  DEFAULT_KV_CACHE_LAYOUT: DEFAULT_KV_CACHE_LAYOUT.
  KVCacheManager._mamba_num_blocks: KVCacheManager#_mamba_num_blocks.
  KVCacheManager.__init__: KVCacheManager#__init__().
  KVCacheManager.get_kv_cache_layout: KVCacheManager#get_kv_cache_layout().
  KVCacheManager.gather_and_reshape: KVCacheManager#gather_and_reshape().
  is_cache_for_ds_v4: is_cache_for_ds_v4().
  is_ds_v4: is_ds_v4().
  KVCacheManager._allocate_mamba: KVCacheManager#_allocate_mamba().
  KVCacheManager._update_layer: KVCacheManager#_update_layer().
  KVCacheManager.get_sharding: KVCacheManager#get_sharding().
---
# Module: [`tpu_inference/runner/kv_cache_manager.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py)

## Classes
### `KVCacheManager`
- def: [`tpu_inference/runner/kv_cache_manager.py:74`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L74)
- signature: `class KVCacheManager:`
- members:
  - `_jitted_gather_continuous_kv_cache(kv_caches: List[jax.Array], start_block, len_block)` — [`L1074`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1074) — JIT-compiled function to gather KV cache slices for all layers at once.
  - `_jitted_gather_kv_cache(kv_caches: List[jax.Array], block_ids: jax.Array)` — [`L1059`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1059) — JIT-compiled function to gather KV cache slices for all layers at once.
  - `_jitted_insert_continuous_kv_cache_from_slice(block_size: int, chunk_size: int, kv_caches: List[jax.Array], kv_cache_slices: List[jax.Array], token_start: int, start_block: int)` — [`L1132`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1132) — JIT-compiled function that dynamically slices the required tokens from KV cache
  - `_jitted_insert_kv_cache(block_size, kv_caches: List[jax.Array], kv_cache_slices: List[jax.Array], block_numbers: List[int])` — [`L1092`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1092) — Iteratively call continuous KV cache insertion for each contiguous block sub-array.
  - `_maybe_set_compact_mamba_num_blocks_override(self, attn_page_size_bytes: int, unpadded_mamba_page_size_bytes: int, num_attn_groups: int, num_mamba_groups: int, num_attn_layers: int, num_mamba_layers: int, group_size: int)` — [`L291`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L291) — Cap mamba layers at `max_num_reqs+1` slots and pin
  - `_update_layer(cache, slices)` — [`L1145`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1145) — The function to apply to each layer's cache and slices.
  - `delete_kv_cache(self)` — [`L995`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L995) — Delete KV cache JAX arrays to free HBM.
  - `gather_and_reshape(layer_kv_cache)` — [`L1066`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1066)
  - `get_kv_cache_for_block_ids(self, block_ids: List[int])` — [`L1164`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1164) — Extracts the KV cache slices for a given list of block IDs.
  - `get_kv_cache_layout(self)` — [`L668`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L668)
  - `get_kv_cache_spec(self)` — [`L442`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L442)
  - `get_sharding(x)` — [`L1219`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1219)
  - `initialize_kv_cache(self, kv_cache_config: KVCacheConfig)` — [`L699`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L699)
  - `insert_request_with_kv_cache(self, request: Request, kv_cache_slices: List[jax.Array], block_ids: List[List[int]])` — [`L1234`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1234) — Inserts a request and its KV cache into the runner. This is used to
  - `maybe_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig)` — [`L672`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L672)
  - `reinitialize_kv_cache(self)` — [`L1032`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1032) — Reinitialize KV cache from the stored configuration.
  - `transfer_kv_cache(self, kv_cache_slices: List[jax.Array])` — [`L1189`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L1189) — Transfers KV cache slices to the runner's mesh.
  - `update_mamba_page_size_padded(self, layers: dict[str, AttentionLayerBase])` — [`L144`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L144) — Pad attention and mamba page sizes so vLLM's num_blocks matches
  - `actual_mamba_num_blocks` — [`L101`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L101)
  - `runner` — [`L77`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L77)
  - `shared_kv_cache_layers` — [`L82`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L82)
  - `use_mla` — [`L83`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L83)
- protocol/private: `__init__`[`L76`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L76), `_allocate_mamba`[`L860`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L860), `_create_attention_spec`[`L103`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L103), `_hybrid_uniform_page_size_bytes`[`L90`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L90), `_mamba_num_blocks`[`L100`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L100)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../layers/common/sharding.md#ShardingAxisName), `mesh`, `input_batch`, `max_num_reqs`, [`add_request`](input_batch.md#InputBatch.add_request), `vllm_config`, `model_config`, `dp_size`, `speculative_config`, `kv_caches`, [`input_batch`](persistent_batch_manager.md#PersistentBatchManager.input_batch), [`logger`](kv_cache_manager.md#logger), `max_model_len`, `persistent_batch_manager`, [`CachedRequestState`](input_batch.md#CachedRequestState), [`output_token_ids`](input_batch.md#CachedRequestState.output_token_ids), `requests`, [`InputBatch`](input_batch.md#InputBatch), `cache_config`, [`create_kv_caches`](kv_cache.md#create_kv_caches), `max_num_tokens`, `TPUModelRunner`, `layer_name_to_kvcache_index`, `block_size`, [`get_attention_page_size_bytes`](kv_cache.md#get_attention_page_size_bytes), [`count`](kv_cache.md#KVCacheMetadata.count), `kv_cache_dtype`, [`compute_kv_share_map`](../models/common/kv_share.md#compute_kv_share_map), [`get_kv_connector_cache_layout`](../offload/utils.md#get_kv_connector_cache_layout), [`generator`](input_batch.md#CachedRequestState.generator), [`shape`](kv_cache.md#KVCacheMetadata.shape), [`KVCacheMetadata`](kv_cache.md#KVCacheMetadata), [`LatencyTracker`](../utils.md#LatencyTracker), [`DEFAULT_KV_CACHE_LAYOUT`](kv_cache_manager.md#DEFAULT_KV_CACHE_LAYOUT), `parallel_config`, [`compute_mtp_kv_share_map`](../models/common/kv_share.md#compute_mtp_kv_share_map), [`dtype`](kv_cache.md#KVCacheMetadata.dtype), [`sharding`](kv_cache.md#KVCacheMetadata.sharding), [`is_cache_for_ds_v4`](kv_cache_manager.md#is_cache_for_ds_v4), [`is_ds_v4`](kv_cache_manager.md#is_ds_v4)
- used by: `use_hybrid_kvcache`, `kv_cache_manager`, [`_precompile_disagg_utils`](compilation_manager.md#CompilationManager._precompile_disagg_utils), `delete_kv_cache`, `get_kv_cache_for_block_ids`, `get_kv_cache_layout`, `get_kv_cache_spec`, `insert_request_with_kv_cache`, `reinitialize_kv_cache`, `transfer_kv_cache`

## Functions
- `is_cache_for_ds_v4(attn_module: AttentionLayerBase)` — [`L63`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L63)
- `is_ds_v4(vllm_config)` — [`L70`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L70)

## Module values
- `DEFAULT_KV_CACHE_LAYOUT` — [`L60`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L60)
- `logger` — [`L56`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L56)

