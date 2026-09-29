---
title: 'Module: easydel/caching/recurrent/cache.py'
type: catalog
provenance: extracted
module: easydel/caching/recurrent/cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching.recurrent.cache`/
symbols:
  RecurrentCacheView: RecurrentCacheView#
  RecurrentCache: RecurrentCache#
  RecurrentCacheConfig: RecurrentCacheConfig#
  RecurrentCacheView.conv_state: RecurrentCacheView#conv_state.
  RecurrentCacheView.recurrent_state: RecurrentCacheView#recurrent_state.
  RecurrentCacheView.concatenate_to_cache: RecurrentCacheView#concatenate_to_cache().
  RecurrentCacheView.init: RecurrentCacheView#init().
  RecurrentCache.views: RecurrentCache#views.
  RecurrentCacheView.reset: RecurrentCacheView#reset().
  RecurrentCache.init_cache: RecurrentCache#init_cache().
  RecurrentCache.to_pure: RecurrentCache#to_pure().
  RecurrentMetadata: RecurrentMetadata#
  RecurrentCache.insert: RecurrentCache#insert().
  RecurrentCacheConfig.create: RecurrentCacheConfig#create().
  RecurrentCacheView.positions: RecurrentCacheView#positions.
  RecurrentCacheView.update_recurrent_state: RecurrentCacheView#update_recurrent_state().
  RecurrentCacheView.__repr__: RecurrentCacheView#__repr__().
  RecurrentCache.update_recurrent_state: RecurrentCache#update_recurrent_state().
  LinearCacheView: LinearCacheView#
  RecurrentCache.update_conv_state: RecurrentCache#update_conv_state().
  RecurrentCache.reset: RecurrentCache#reset().
  RecurrentCache.update_seq: RecurrentCache#update_seq().
  RecurrentCache.from_pure: RecurrentCache#from_pure().
  RecurrentCacheConfig.create_for_mamba: RecurrentCacheConfig#create_for_mamba().
  RecurrentCacheView.metadata: RecurrentCacheView#metadata.
  LinearMetadata: LinearMetadata#
  RecurrentCacheConfig.create_for_mamba2: RecurrentCacheConfig#create_for_mamba2().
  RecurrentCacheView.update_conv_state: RecurrentCacheView#update_conv_state().
  RecurrentCache.init_empty: RecurrentCache#init_empty().
  RecurrentCacheView.update_ssm_state: RecurrentCacheView#update_ssm_state().
  RecurrentCache.update_ssm_state: RecurrentCache#update_ssm_state().
  LinearCache: LinearCache#
  LinearCacheConfig: LinearCacheConfig#
  RecurrentCache.__repr__: RecurrentCache#__repr__().
  RecurrentCacheConfig.batch_size: RecurrentCacheConfig#batch_size.
  RecurrentCacheView.seqlen_offset: RecurrentCacheView#seqlen_offset.
  RecurrentCacheView.layer_index: RecurrentCacheView#layer_index.
  RecurrentCacheView.ssm_states: RecurrentCacheView#ssm_states().
  RecurrentCacheView.conv_states: RecurrentCacheView#conv_states().
  RecurrentCacheView.__str__: RecurrentCacheView#__str__.
  RecurrentCache.__str__: RecurrentCache#__str__.
  RecurrentCacheConfig.conv_kernel_size: RecurrentCacheConfig#conv_kernel_size.
  RecurrentCacheConfig.num_hidden_layers: RecurrentCacheConfig#num_hidden_layers.
  RecurrentCacheConfig.partition_axis: RecurrentCacheConfig#partition_axis.
  RecurrentCacheConfig.conv_dim: RecurrentCacheConfig#conv_dim.
  RecurrentCacheConfig.recurrent_state_shape: RecurrentCacheConfig#recurrent_state_shape.
  RecurrentCacheConfig.seqlen_offset: RecurrentCacheConfig#seqlen_offset.
---
# Module: [`easydel/caching/recurrent/cache.py`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py)

## Classes
### `LinearCache`  ·  implements/extends RecurrentCache
- def: [`easydel/caching/recurrent/cache.py:715`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L715)
- signature: `class LinearCache(RecurrentCache):`
- uses (calls/refs, reference-scoped): [`RecurrentCache`](cache.md#RecurrentCache)
- used by: [`RecurrentCache`](cache.md#RecurrentCache)

### `LinearCacheConfig`  ·  implements/extends RecurrentCacheConfig
- def: [`easydel/caching/recurrent/cache.py:719`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L719)
- signature: `class LinearCacheConfig(RecurrentCacheConfig):`
- uses (calls/refs, reference-scoped): [`RecurrentCacheConfig`](cache.md#RecurrentCacheConfig)
- used by: [`RecurrentCacheConfig`](cache.md#RecurrentCacheConfig)

### `LinearCacheView`  ·  implements/extends RecurrentCacheView
- def: [`easydel/caching/recurrent/cache.py:723`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L723)
- signature: `class LinearCacheView(RecurrentCacheView):`
- uses (calls/refs, reference-scoped): [`RecurrentCacheView`](cache.md#RecurrentCacheView)
- used by: [`__call__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextLinearAttention.__call__), [`RecurrentCacheView`](cache.md#RecurrentCacheView), [`forward`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextFullAttention.forward), [`__call__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextDecoderLayer.__call__)

### `LinearMetadata`  ·  implements/extends RecurrentMetadata
- def: [`easydel/caching/recurrent/cache.py:727`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L727)
- signature: `class LinearMetadata(RecurrentMetadata):`
- uses (calls/refs, reference-scoped): [`RecurrentMetadata`](cache.md#RecurrentMetadata)
- used by: [`__call__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextLinearAttention.__call__), [`RecurrentMetadata`](cache.md#RecurrentMetadata)

### `RecurrentCache`  ·  implements/extends BaseCache
- def: [`easydel/caching/recurrent/cache.py:450`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L450)
- doc: Multi-layer cache container for recurrent models.
- signature: `class RecurrentCache(BaseCache):`
- members:
  - `from_pure(cls, cache_data: list[dict[str, tp.Any] | None], metadata: RecurrentCacheConfig | None = None)` — [`L619`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L619) — Reconstruct cache from pure Python data structure.
  - `init_cache(cls, config: RecurrentCacheConfig, dtype: jnp.dtype | None = None, partition_specs: PartitionSpec | None = None)` — [`L463`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L463) — Initialize a complete recurrent cache with views for all layers.
  - `init_empty(cls, num_hidden_layers: int)` — [`L566`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L566) — Initialize an empty cache without allocated storage.
  - `insert(self, other: RecurrentCache, slot: int)` — [`L654`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L654) — Insert another cache's contents at specified batch slot.
  - `reset(self)` — [`L556`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L556) — Reset all cache layers to zero states.
  - `to_pure(self)` — [`L589`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L589) — Convert cache to pure Python data structure for serialization.
  - `update_conv_state(self, layer_idx: int, new_conv_state: Float[Array, "batch conv_dim"], cache_position: Int[Array, ...] | None = None)` — [`L494`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L494) — Update convolutional state for a specific layer.
  - `update_recurrent_state(self, layer_idx: int, new_recurrent_state: Float[Array, "batch ..."])` — [`L522`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L522) — Update recurrent state for a specific layer.
  - `update_seq(self, num: int)` — [`L577`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L577) — Update sequence positions across all layers.
  - `update_ssm_state(self, layer_idx: int, new_ssm_state: Float[Array, "batch ..."])` — [`L548`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L548) — Alias for update_recurrent_state (backward compatibility).
  - `views` — [`L460`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L460)
- protocol/private: `__repr__`[`L697`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L697), `__str__`[`L700`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L700)
- uses (calls/refs, reference-scoped): [`RecurrentCacheView`](cache.md#RecurrentCacheView), [`BaseCache`](../_abstracts.md#BaseCache), [`RecurrentCacheConfig`](cache.md#RecurrentCacheConfig), [`conv_state`](cache.md#RecurrentCacheView.conv_state), [`recurrent_state`](cache.md#RecurrentCacheView.recurrent_state), [`init`](cache.md#RecurrentCacheView.init), [`reset`](cache.md#RecurrentCacheView.reset), [`positions`](cache.md#RecurrentCacheView.positions), [`update_recurrent_state`](cache.md#RecurrentCacheView.update_recurrent_state), [`metadata`](cache.md#RecurrentCacheView.metadata), [`update_conv_state`](cache.md#RecurrentCacheView.update_conv_state), [`LinearCache`](cache.md#LinearCache), [`seqlen_offset`](cache.md#RecurrentCacheView.seqlen_offset), [`num_hidden_layers`](cache.md#RecurrentCacheConfig.num_hidden_layers)
- used by: [`BaseModuleProtocol`](../../infra/mixins/protocol.md#BaseModuleProtocol), [`BaseCache`](../_abstracts.md#BaseCache), [`__call__`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaModel.__call__), [`get_required_cache_class`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_required_cache_class), [`RecurrentState`](../../infra/mixins/protocol.md#RecurrentState), [`__call__`](../../modules/mamba/modeling_mamba.md#MambaModel.__call__), [`__call__`](../../modules/mamba2/modeling_mamba2.md#Mamba2Model.__call__), [`init_cache`](../_abstracts.md#BaseCache.init_cache), [`init_empty`](../_abstracts.md#BaseCache.init_empty), [`prepare_inputs_for_generation`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaForCausalLM.prepare_inputs_for_generation), [`prepare_inputs_for_generation`](../../modules/mamba/modeling_mamba.md#MambaForCausalLM.prepare_inputs_for_generation), [`prepare_inputs_for_generation`](../../modules/mamba2/modeling_mamba2.md#Mamba2ForCausalLM.prepare_inputs_for_generation), [`__call__`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaForCausalLM.__call__), [`__call__`](../../modules/mamba/modeling_mamba.md#MambaForCausalLM.__call__), [`__call__`](../../modules/mamba2/modeling_mamba2.md#Mamba2ForCausalLM.__call__), [`LinearCache`](cache.md#LinearCache), [`cache`](../../modules/mamba/modeling_mamba.md#MambaCausalLMOutput.cache), [`cache`](../../modules/mamba/modeling_mamba.md#MambaOutput.cache), [`cache_params`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaCausalLMOutput.cache_params), [`cache_params`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaOutput.cache_params), [`cache_params`](../../modules/mamba2/modeling_mamba2.md#Mamba2CausalLMOutput.cache_params), [`cache_params`](../../modules/mamba2/modeling_mamba2.md#Mamba2Output.cache_params), [`past_key_values`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaCausalLMOutput.past_key_values), [`past_key_values`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaOutput.past_key_values)

### `RecurrentCacheConfig`  ·  implements/extends BaseCacheConfig
- def: [`easydel/caching/recurrent/cache.py:47`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L47)
- doc: Metadata for unified recurrent cache configuration.
- signature: `class RecurrentCacheConfig(BaseCacheConfig):`
- members:
  - `create(cls, num_hidden_layers: int, partition_axis: es.PartitionAxis, batch_size: int, conv_dim: int, conv_kernel_size: int, recurrent_state_shape: tuple[int, ...], seqlen_offset: int = 0)` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L80) — Create a RecurrentCacheConfig instance with validation.
  - `create_for_mamba(cls, num_hidden_layers: int, partition_axis: es.PartitionAxis, batch_size: int, intermediate_size: int, ssm_state_size: int, conv_kernel_size: int)` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L133) — Create metadata for Mamba (single-head SSM) models.
  - `create_for_mamba2(cls, num_hidden_layers: int, partition_axis: es.PartitionAxis, batch_size: int, intermediate_size: int, num_heads: int, head_dim: int, state_size: int, conv_kernel_size: int, n_groups: int = 1)` — [`L169`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L169) — Create metadata for Mamba2 (multi-head SSM) models.
  - `batch_size` — [`L73`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L73)
  - `conv_dim` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L74)
  - `conv_kernel_size` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L75)
  - `num_hidden_layers` — [`L71`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L71)
  - `partition_axis` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L72)
  - `recurrent_state_shape` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L76)
  - `seqlen_offset` — [`L77`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L77)
- uses (calls/refs, reference-scoped): [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig), [`LinearCacheConfig`](cache.md#LinearCacheConfig)
- used by: [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig), [`concatenate_to_cache`](cache.md#RecurrentCacheView.concatenate_to_cache), [`init`](cache.md#RecurrentCacheView.init), [`create`](../_abstracts.md#BaseCacheConfig.create), [`prepare_inputs_for_generation`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaForCausalLM.prepare_inputs_for_generation), [`prepare_inputs_for_generation`](../../modules/mamba/modeling_mamba.md#MambaForCausalLM.prepare_inputs_for_generation), [`prepare_inputs_for_generation`](../../modules/mamba2/modeling_mamba2.md#Mamba2ForCausalLM.prepare_inputs_for_generation), [`init_cache`](cache.md#RecurrentCache.init_cache), [`to_pure`](cache.md#RecurrentCache.to_pure), [`create_recurrent_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.create_recurrent_cache_config), [`create_recurrent_cache_config`](../../modules/minimax/modeling_minimax.md#MiniMaxForCausalLM.create_recurrent_cache_config), [`from_pure`](cache.md#RecurrentCache.from_pure), [`metadata`](cache.md#RecurrentCacheView.metadata), [`LinearCacheConfig`](cache.md#LinearCacheConfig)

### `RecurrentCacheView`  ·  implements/extends BaseCacheView
- def: [`easydel/caching/recurrent/cache.py:214`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L214)
- doc: Single-layer cache view for recurrent state management.
- signature: `class RecurrentCacheView(BaseCacheView):`
- members:
  - `concatenate_to_cache(self, conv_state: Float[Array, "batch conv_dim"] | None = None, recurrent_state: Float[Array, "batch ..."] | None = None, cache_position: Int[Array, ...] | None = None)` — [`L303`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L303) — Update cache state with new conv and/or recurrent states.
  - `conv_states(self)` — [`L417`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L417) — Alias for conv_state (backward compatibility with MambaCache).
  - `init(cls, config: RecurrentCacheConfig, layer_index: int | None = None, *, dtype: jnp.dtype = jnp.bfloat16, partition_specs: PartitionSpec | None = None)` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L248) — Initialize a RecurrentCacheView from a cache config.
  - `reset(self)` — [`L421`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L421) — Reset all cache states to zeros.
  - `ssm_states(self)` — [`L412`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L412) — Alias for recurrent_state (backward compatibility with MambaCache).
  - `update_conv_state(self, new_conv_state: Float[Array, "batch conv_dim"], cache_position: Int[Array, ...] | None = None)` — [`L362`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L362) — Update the convolutional state with new values.
  - `update_recurrent_state(self, new_recurrent_state: Float[Array, "batch ..."])` — [`L385`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L385) — Update the recurrent (SSM) state.
  - `update_ssm_state(self, new_ssm_state: Float[Array, "batch ..."])` — [`L404`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L404) — Alias for update_recurrent_state (backward compatibility).
  - `conv_state` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L237)
  - `layer_index` — [`L245`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L245)
  - `metadata` — [`L244`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L244)
  - `positions` — [`L239`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L239)
  - `recurrent_state` — [`L238`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L238)
  - `seqlen_offset` — [`L243`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L243)
- protocol/private: `__repr__`[`L436`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L436), `__str__`[`L446`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L446)
- uses (calls/refs, reference-scoped): [`BaseCacheView`](../_abstracts.md#BaseCacheView), [`RecurrentCacheConfig`](cache.md#RecurrentCacheConfig), [`LinearCacheView`](cache.md#LinearCacheView), [`batch_size`](cache.md#RecurrentCacheConfig.batch_size), [`conv_kernel_size`](cache.md#RecurrentCacheConfig.conv_kernel_size), [`conv_dim`](cache.md#RecurrentCacheConfig.conv_dim), [`partition_axis`](cache.md#RecurrentCacheConfig.partition_axis), [`recurrent_state_shape`](cache.md#RecurrentCacheConfig.recurrent_state_shape), [`seqlen_offset`](cache.md#RecurrentCacheConfig.seqlen_offset)
- used by: [`__call__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextLinearAttention.__call__), [`__call__`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaMixer.__call__), [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`__call__`](../../modules/mamba2/modeling_mamba2.md#Mamba2Mixer.__call__), [`__call__`](../../modules/mamba/modeling_mamba.md#MambaMixer.__call__), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxDecoderLayer.__call__), [`get_requirements`](../../operations/kernels/gated_delta_rule.md#GatedDeltaRuleOp.get_requirements), [`get_requirements`](../../operations/kernels/ssm1.md#SSM1Op.get_requirements), [`get_requirements`](../../operations/kernels/ssm2.md#SSM2Op.get_requirements), [`views`](../hybrid/cache.md#HybridCache.views), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxLightningAttention.__call__), [`BaseCacheView`](../_abstracts.md#BaseCacheView), [`get_operations_cache_view`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_view), [`view`](../hybrid/cache.md#view), [`insert`](../hybrid/cache.md#HybridCache.insert), [`get_required_cache_class`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_required_cache_class), [`init`](../_abstracts.md#BaseCacheView.init), [`clear_recurrent_slots`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.clear_recurrent_slots), [`concatenate_to_cache`](../_abstracts.md#BaseCacheView.concatenate_to_cache), [`views`](cache.md#RecurrentCache.views), [`new_recurrent_state`](../hybrid/cache.md#new_recurrent_state), [`__call__`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaBlock.__call__), [`concatenate_to_cache`](../hybrid/cache.md#ParallelHybridCacheView.concatenate_to_cache), [`from_pure`](../hybrid/cache.md#HybridCache.from_pure), [`to_pure`](../hybrid/cache.md#HybridCache.to_pure), [`recurrent`](../hybrid/cache.md#ParallelHybridCacheView.recurrent), [`get_view`](../hybrid/cache.md#HybridCache.get_view), [`init_cache`](cache.md#RecurrentCache.init_cache), [`__call__`](../../modules/mamba/modeling_mamba.md#MambaBlock.__call__), [`__call__`](../../modules/mamba2/modeling_mamba2.md#Mamba2Block.__call__), [`to_pure`](cache.md#RecurrentCache.to_pure), [`insert`](cache.md#RecurrentCache.insert), [`LinearCacheView`](cache.md#LinearCacheView), [`update_recurrent_state`](../hybrid/cache.md#HybridCache.update_recurrent_state), [`update_recurrent_state`](cache.md#RecurrentCache.update_recurrent_state), [`from_pure`](cache.md#RecurrentCache.from_pure), [`get_operations_cache_view`](../../modules/minimax/modeling_minimax.md#MiniMaxForCausalLM.get_operations_cache_view), [`reset`](cache.md#RecurrentCache.reset), [`update_conv_state`](cache.md#RecurrentCache.update_conv_state)  (+1 more)

### `RecurrentMetadata`  ·  implements/extends BaseRunTimeMetadata
- def: [`easydel/caching/recurrent/cache.py:703`](../../../../../../../raw/code/EasyDeL/easydel/caching/recurrent/cache.py#L703)
- doc: Runtime metadata for recurrent cache operations.
- signature: `class RecurrentMetadata(BaseRunTimeMetadata):`
- uses (calls/refs, reference-scoped): [`BaseRunTimeMetadata`](../_abstracts.md#BaseRunTimeMetadata), [`LinearMetadata`](cache.md#LinearMetadata)
- used by: [`build`](../_metadatabuilder.md#AttentionMetadataBuilder.build), [`BaseRunTimeMetadata`](../_abstracts.md#BaseRunTimeMetadata), [`LinearMetadata`](cache.md#LinearMetadata), [`for_recurrent`](../_abstracts.md#OperationsMetadata.for_recurrent), [`build_recurrent_metadata`](../_metadatabuilder.md#AttentionMetadataBuilder.build_recurrent_metadata)

