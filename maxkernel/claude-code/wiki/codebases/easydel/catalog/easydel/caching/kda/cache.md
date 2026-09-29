---
title: 'Module: easydel/caching/kda/cache.py'
type: catalog
provenance: extracted
module: easydel/caching/kda/cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching.kda.cache`/KDA
symbols:
  KDACacheView: CacheView#
  KDACache.insert: Cache#insert().
  KDACache.to_pure: Cache#to_pure().
  KDACacheView.concatenate_to_cache: CacheView#concatenate_to_cache().
  KDACacheView.reset: CacheView#reset().
  KDACacheView.init: CacheView#init().
  KDACacheConfig: CacheConfig#
  KDACacheView.__repr__: CacheView#__repr__().
  KDACacheView.recurrent_state: CacheView#recurrent_state.
  KDACacheView.positions: CacheView#positions.
  KDACache: Cache#
  KDACache.init_cache: Cache#init_cache().
  KDACache.views: Cache#views.
  KDACache.reset: Cache#reset().
  KDACache.from_pure: Cache#from_pure().
  KDACacheView.q_conv_state: CacheView#q_conv_state.
  KDACacheView.k_conv_state: CacheView#k_conv_state.
  KDACacheView.v_conv_state: CacheView#v_conv_state.
  KDACacheView.metadata: CacheView#metadata.
  KDAMetadata: Metadata#
  KDACacheView.update_kda_states: CacheView#update_kda_states().
  KDACacheConfig.batch_size: CacheConfig#batch_size.
  KDACacheView.layer_index: CacheView#layer_index.
  KDACacheConfig.create: CacheConfig#create().
  KDACache.__repr__: Cache#__repr__().
  KDACacheConfig.d_conv: CacheConfig#d_conv.
  KDACacheView.__str__: CacheView#__str__.
  KDACache.init_empty: Cache#init_empty().
  KDACache.__str__: Cache#__str__.
  KDACacheConfig.key_dim: CacheConfig#key_dim.
  KDACacheConfig.num_hidden_layers: CacheConfig#num_hidden_layers.
  KDACacheConfig.partition_axis: CacheConfig#partition_axis.
  KDACacheConfig.value_dim: CacheConfig#value_dim.
  KDACacheConfig.recurrent_state_shape: CacheConfig#recurrent_state_shape.
---
# Module: [`easydel/caching/kda/cache.py`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py)

## Classes
### `KDACache`  ·  implements/extends BaseCache
- def: [`easydel/caching/kda/cache.py:336`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L336)
- doc: Multi-layer KDA cache container.
- signature: `class KDACache(BaseCache):`
- members:
  - `from_pure(cls, cache_data: list[dict[str, tp.Any]], metadata: KDACacheConfig | None = None)` — [`L447`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L447) — Reconstruct cache from pure Python data.
  - `init_cache(cls, config: KDACacheConfig, dtype: jnp.dtype | None = None, partition_specs: PartitionSpec | None = None)` — [`L355`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L355) — Initialize a complete KDA cache for all layers.
  - `init_empty(cls, num_hidden_layers: int)` — [`L389`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L389) — Initialize an empty KDA cache with None views.
  - `insert(self, other: KDACache, slot: int)` — [`L480`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L480) — Insert another cache's contents at a specific batch slot.
  - `reset(self)` — [`L402`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L402) — Reset all layer caches to zeros.
  - `to_pure(self)` — [`L416`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L416) — Convert cache to pure Python data for serialization.
  - `views` — [`L352`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L352)
- protocol/private: `__repr__`[`L413`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L413), `__str__`[`L520`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L520)
- uses (calls/refs, reference-scoped): [`BaseCache`](../_abstracts.md#BaseCache), [`KDACacheView`](cache.md#KDACacheView), [`reset`](cache.md#KDACacheView.reset), [`init`](cache.md#KDACacheView.init), [`KDACacheConfig`](cache.md#KDACacheConfig), [`positions`](cache.md#KDACacheView.positions), [`recurrent_state`](cache.md#KDACacheView.recurrent_state), [`k_conv_state`](cache.md#KDACacheView.k_conv_state), [`q_conv_state`](cache.md#KDACacheView.q_conv_state), [`v_conv_state`](cache.md#KDACacheView.v_conv_state), [`metadata`](cache.md#KDACacheView.metadata), [`layer_index`](cache.md#KDACacheView.layer_index), [`num_hidden_layers`](cache.md#KDACacheConfig.num_hidden_layers)
- used by: [`BaseCache`](../_abstracts.md#BaseCache), [`init_cache`](../_abstracts.md#BaseCache.init_cache), [`init_empty`](../_abstracts.md#BaseCache.init_empty)

### `KDACacheConfig`  ·  implements/extends BaseCacheConfig
- def: [`easydel/caching/kda/cache.py:65`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L65)
- doc: Configuration for KDA (Key-Driven Attention) cache.
- signature: `class KDACacheConfig(BaseCacheConfig):`
- members:
  - `create(cls, num_hidden_layers: int, partition_axis: PartitionAxis, batch_size: int, key_dim: int, value_dim: int, d_conv: int, recurrent_state_shape: tuple[int, ...])` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L108) — Create and validate a KDACacheConfig.
  - `batch_size` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L101)
  - `d_conv` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L104)
  - `key_dim` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L102)
  - `num_hidden_layers` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L99)
  - `partition_axis` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L100)
  - `recurrent_state_shape` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L105)
  - `value_dim` — [`L103`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L103)
- uses (calls/refs, reference-scoped): [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig)
- used by: [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig), [`to_pure`](cache.md#KDACache.to_pure), [`init`](cache.md#KDACacheView.init), [`create`](../_abstracts.md#BaseCacheConfig.create), [`init_cache`](cache.md#KDACache.init_cache), [`create_kda_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.create_kda_cache_config), [`from_pure`](cache.md#KDACache.from_pure), [`metadata`](cache.md#KDACacheView.metadata)

### `KDACacheView`  ·  implements/extends BaseCacheView
- def: [`easydel/caching/kda/cache.py:160`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L160)
- doc: Single-layer KDA cache view.
- signature: `class KDACacheView(BaseCacheView):`
- members:
  - `concatenate_to_cache(self, q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None, k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None, v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None, recurrent_state: Float[Array, "batch ..."] | None = None)` — [`L238`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L238) — Update cache with new Q/K/V conv states and/or recurrent state.
  - `init(cls, config: KDACacheConfig, layer_index: int | None = None, *, dtype: jnp.dtype = jnp.bfloat16, partition_specs: PartitionSpec | None = None)` — [`L184`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L184) — Initialize a KDACacheView from a cache config.
  - `reset(self)` — [`L306`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L306) — Reset cache to zeros.
  - `update_kda_states(self, new_q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None, new_k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None, new_v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None, new_recurrent_state: Float[Array, "batch ..."] | None = None)` — [`L277`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L277) — Update KDA states with new values.
  - `k_conv_state` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L176)
  - `layer_index` — [`L181`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L181)
  - `metadata` — [`L180`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L180)
  - `positions` — [`L179`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L179)
  - `q_conv_state` — [`L175`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L175)
  - `recurrent_state` — [`L178`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L178)
  - `v_conv_state` — [`L177`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L177)
- protocol/private: `__repr__`[`L325`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L325), `__str__`[`L332`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L332)
- uses (calls/refs, reference-scoped): [`BaseCacheView`](../_abstracts.md#BaseCacheView), [`KDACacheConfig`](cache.md#KDACacheConfig), [`batch_size`](cache.md#KDACacheConfig.batch_size), [`d_conv`](cache.md#KDACacheConfig.d_conv), [`key_dim`](cache.md#KDACacheConfig.key_dim), [`partition_axis`](cache.md#KDACacheConfig.partition_axis), [`recurrent_state_shape`](cache.md#KDACacheConfig.recurrent_state_shape), [`value_dim`](cache.md#KDACacheConfig.value_dim)
- used by: [`__call__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiDeltaAttention.__call__), [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`get_requirements`](../../operations/kernels/kda.md#KernelDeltaAttnOp.get_requirements), [`views`](../hybrid/cache.md#HybridCache.views), [`__call__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiDecoderLayer.__call__), [`BaseCacheView`](../_abstracts.md#BaseCacheView), [`view`](../hybrid/cache.md#view), [`insert`](../hybrid/cache.md#HybridCache.insert), [`init`](../_abstracts.md#BaseCacheView.init), [`insert`](cache.md#KDACache.insert), [`to_pure`](cache.md#KDACache.to_pure), [`concatenate_to_cache`](../_abstracts.md#BaseCacheView.concatenate_to_cache), [`new_recurrent_state`](../hybrid/cache.md#new_recurrent_state), [`from_pure`](../hybrid/cache.md#HybridCache.from_pure), [`to_pure`](../hybrid/cache.md#HybridCache.to_pure), [`get_view`](../hybrid/cache.md#HybridCache.get_view), [`init_cache`](cache.md#KDACache.init_cache), [`update_kv_cache`](../hybrid/cache.md#HybridCache.update_kv_cache), [`from_pure`](cache.md#KDACache.from_pure), [`reset`](../hybrid/cache.md#HybridCache.reset), [`reset`](cache.md#KDACache.reset), [`views`](cache.md#KDACache.views), [`get_cache_position`](../hybrid/cache.md#HybridCache.get_cache_position)

### `KDAMetadata`  ·  implements/extends BaseRunTimeMetadata
- def: [`easydel/caching/kda/cache.py:523`](../../../../../../../raw/code/EasyDeL/easydel/caching/kda/cache.py#L523)
- doc: Runtime metadata for KDA cache operations.
- signature: `class KDAMetadata(BaseRunTimeMetadata):`
- uses (calls/refs, reference-scoped): [`BaseRunTimeMetadata`](../_abstracts.md#BaseRunTimeMetadata)
- used by: [`__call__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiDeltaAttention.__call__), [`BaseRunTimeMetadata`](../_abstracts.md#BaseRunTimeMetadata)

