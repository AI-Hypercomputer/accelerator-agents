---
title: 'Module: easydel/caching/lightning/cache.py'
type: catalog
provenance: extracted
module: easydel/caching/lightning/cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching.lightning.cache`/Lightning
symbols:
  LightningCacheView: CacheView#
  LightningCache.insert: Cache#insert().
  LightningCacheConfig: CacheConfig#
  LightningCache.to_pure: Cache#to_pure().
  LightningCache: Cache#
  LightningCache.init_cache: Cache#init_cache().
  LightningCacheView.key_value: CacheView#key_value.
  LightningCache.from_pure: Cache#from_pure().
  LightningCache.views: Cache#views.
  LightningCacheView.init: CacheView#init().
  LightningCacheView.__repr__: CacheView#__repr__().
  LightningCacheView.metadata: CacheView#metadata.
  LightningMetadata: Metadata#
  LightningCacheConfig.create: CacheConfig#create().
  LightningCache.__repr__: Cache#__repr__().
  LightningCacheView.layer_index: CacheView#layer_index.
  LightningCacheView.concatenate_to_cache: CacheView#concatenate_to_cache().
  LightningCacheView.__str__: CacheView#__str__.
  LightningCache.init_empty: Cache#init_empty().
  LightningCache.__str__: Cache#__str__.
  LightningCacheConfig.partition_axis: CacheConfig#partition_axis.
  LightningCacheConfig.batch_size: CacheConfig#batch_size.
  LightningCacheConfig.num_heads: CacheConfig#num_heads.
  LightningCacheConfig.head_dim: CacheConfig#head_dim.
  LightningCacheConfig.key_heads: CacheConfig#key_heads.
  LightningCacheConfig.value_heads: CacheConfig#value_heads.
  LightningCacheConfig.key_dim: CacheConfig#key_dim.
  LightningCacheConfig.value_dim: CacheConfig#value_dim.
---
# Module: [`easydel/caching/lightning/cache.py`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py)

## Classes
### `LightningCache`  ·  implements/extends BaseCache
- def: [`easydel/caching/lightning/cache.py:249`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L249)
- doc: Multi-layer Lightning attention cache container.
- signature: `class LightningCache(BaseCache):`
- members:
  - `from_pure(cls, cache_data: list[dict[str, tp.Any]], metadata: LightningCacheConfig | None = None)` — [`L336`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L336) — Reconstruct cache from pure Python data.
  - `init_cache(cls, num_hidden_layers: int, config: LightningCacheConfig, dtype: jnp.dtype | None = None)` — [`L264`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L264) — Initialize Lightning cache for all model layers.
  - `init_empty(cls, num_hidden_layers: int)` — [`L292`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L292) — Initialize empty Lightning cache structure.
  - `insert(self, other: LightningCache, slot: int)` — [`L365`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L365) — Insert another cache's contents at a specific batch slot.
  - `to_pure(self)` — [`L309`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L309) — Convert cache to pure Python data for serialization.
  - `views` — [`L261`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L261)
- protocol/private: `__repr__`[`L306`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L306), `__str__`[`L400`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L400)
- uses (calls/refs, reference-scoped): [`BaseCache`](../_abstracts.md#BaseCache), [`LightningCacheView`](cache.md#LightningCacheView), [`LightningCacheConfig`](cache.md#LightningCacheConfig), [`key_value`](cache.md#LightningCacheView.key_value), [`init`](cache.md#LightningCacheView.init), [`metadata`](cache.md#LightningCacheView.metadata), [`layer_index`](cache.md#LightningCacheView.layer_index)
- used by: [`BaseCache`](../_abstracts.md#BaseCache), [`init_cache`](../_abstracts.md#BaseCache.init_cache), [`init_empty`](../_abstracts.md#BaseCache.init_empty)

### `LightningCacheConfig`  ·  implements/extends BaseCacheConfig
- def: [`easydel/caching/lightning/cache.py:68`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L68)
- doc: Metadata configuration for Lightning attention cache.
- signature: `class LightningCacheConfig(BaseCacheConfig):`
- members:
  - `create(cls, partition_axis: es.PartitionAxis, batch_size: int | None = None, num_heads: int | None = None, head_dim: int | None = None, key_heads: int | None = None, value_heads: int | None = None, key_dim: int | None = None, value_dim: int | None = None)` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L104) — Create and validate Lightning cache metadata.
  - `batch_size` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L95)
  - `head_dim` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L97)
  - `key_dim` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L100)
  - `key_heads` — [`L98`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L98)
  - `num_heads` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L96)
  - `partition_axis` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L94)
  - `value_dim` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L101)
  - `value_heads` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L99)
- uses (calls/refs, reference-scoped): [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig)
- used by: [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig), [`create`](../_abstracts.md#BaseCacheConfig.create), [`to_pure`](cache.md#LightningCache.to_pure), [`init_cache`](cache.md#LightningCache.init_cache), [`create_lightning_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.create_lightning_cache_config), [`from_pure`](cache.md#LightningCache.from_pure), [`init`](cache.md#LightningCacheView.init), [`metadata`](cache.md#LightningCacheView.metadata)

### `LightningCacheView`  ·  implements/extends BaseCacheView
- def: [`easydel/caching/lightning/cache.py:158`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L158)
- doc: Single-layer cache view for Lightning attention.
- signature: `class LightningCacheView(BaseCacheView):`
- members:
  - `concatenate_to_cache(self, query: Float[Array, "batch query_len num_heads head_dim"], key: Float[Array, "batch query_len num_key_heads key_dim"], value: Float[Array, "batch query_len num_value_heads value_dim"], attention_mask: Bool[Array, "batch 1 query_len seq_len"] | Float[Array, "batch 1 query_len seq_len"], kv_sharding: PartitionSpec, quantizer: EasyQuantizer, causal_mask: Bool[Array, "batch 1 query_len seq_len"] | bool | None = None, token_type_ids: Int[Array, "batch query_len"] | None = None)` — [`L211`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L211) — Not implemented for Lightning attention.
  - `init(cls, config: LightningCacheConfig, layer_index: int | None = None, *, dtype: jnp.dtype = jnp.bfloat16, partition_specs: PartitionSpec | None = None)` — [`L181`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L181) — Initialize a Lightning cache view for a single layer.
  - `key_value` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L176)
  - `layer_index` — [`L178`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L178)
  - `metadata` — [`L177`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L177)
- protocol/private: `__repr__`[`L241`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L241), `__str__`[`L245`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L245)
- uses (calls/refs, reference-scoped): [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`BaseCacheView`](../_abstracts.md#BaseCacheView), [`LightningCacheConfig`](cache.md#LightningCacheConfig)
- used by: [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`BaseCacheView`](../_abstracts.md#BaseCacheView), [`init`](../_abstracts.md#BaseCacheView.init), [`concatenate_to_cache`](../_abstracts.md#BaseCacheView.concatenate_to_cache), [`insert`](cache.md#LightningCache.insert), [`to_pure`](cache.md#LightningCache.to_pure), [`init_cache`](cache.md#LightningCache.init_cache), [`from_pure`](cache.md#LightningCache.from_pure), [`views`](cache.md#LightningCache.views)

### `LightningMetadata`  ·  implements/extends BaseRunTimeMetadata
- def: [`easydel/caching/lightning/cache.py:403`](../../../../../../../raw/code/EasyDeL/easydel/caching/lightning/cache.py#L403)
- doc: Runtime metadata for Lightning attention cache operations.
- signature: `class LightningMetadata(BaseRunTimeMetadata):`
- uses (calls/refs, reference-scoped): [`BaseRunTimeMetadata`](../_abstracts.md#BaseRunTimeMetadata)
- used by: [`BaseRunTimeMetadata`](../_abstracts.md#BaseRunTimeMetadata)

