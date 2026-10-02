---
title: 'Module: axlearn/common/kv_cache/base_kv_cache.py'
type: catalog
provenance: extracted
module: axlearn/common/kv_cache/base_kv_cache.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kv_cache.base_kv_cache`/
symbols:
  KVState: KVState#
  BaseKVCache: BaseKVCache#
  KVState.k_proj: KVState#k_proj.
  KVState.v_proj: KVState#v_proj.
  KVState.key_positions: KVState#key_positions.
  BaseKVCache.init_states: BaseKVCache#init_states().
  BaseKVCache.as_dense_kv: BaseKVCache#as_dense_kv().
  BaseKVCache.extend_step: BaseKVCache#extend_step().
  BaseKVCache.Output: BaseKVCache#Output#
  BaseKVCache.Config: BaseKVCache#Config#
  KVState.page_indices: KVState#page_indices.
  BaseKVCache._cache_dtype: BaseKVCache#_cache_dtype().
  BaseKVCache.Shape: BaseKVCache#Shape#
  BaseKVCache.Shape.batch_size: BaseKVCache#Shape#batch_size.
  BaseKVCache.Shape.num_kv_heads: BaseKVCache#Shape#num_kv_heads.
  BaseKVCache.Shape.per_head_dim: BaseKVCache#Shape#per_head_dim.
  BaseKVCache.Config.kv_partition_spec: BaseKVCache#Config#kv_partition_spec.
  BaseKVCache.Shape.kv_len: BaseKVCache#Shape#kv_len.
  BaseKVCache.Config.cache_dtype: BaseKVCache#Config#cache_dtype.
---
# Module: [`axlearn/common/kv_cache/base_kv_cache.py`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py)

## Classes
### `BaseKVCache`  ·  implements/extends BaseLayer
- def: [`axlearn/common/kv_cache/base_kv_cache.py:31`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L31) — documented in [axlearn-common-kv_cache-base_kv_cache](../../../../concepts/axlearn-common-kv_cache-base_kv_cache.md)
- doc: Abstract base class for KV cache.
- signature: `class BaseKVCache(BaseLayer):`
- members:
  - `as_dense_kv(cls, kv_state: KVState)` — [`L118`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L118) — Return a dense `(k_proj, v_proj)` pair from a dense `KVState`.
  - `extend_step(self, cached_states: Nested[Tensor], *, k_proj: Tensor, v_proj: Tensor, key_positions: Tensor, segment_ids: Optional[Tensor] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L83`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L83) — Updates the KV cache per extend step.
  - `init_states(self, shape: Shape, *, dtype: jnp.dtype)` — [`L71`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L71) — Initializes KV cache.
- protocol/private: `_cache_dtype`[`L64`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L64)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`config`](../config.md#Configurable.config), [`BaseLayer`](../base_layer.md#BaseLayer), [`KVState`](base_kv_cache.md#KVState), [`KVCache`](kv_cache.md#KVCache), [`init_states`](kv_cache.md#KVCache.init_states), [`nowrap`](../module.md#nowrap), [`init_states`](sliding_window_kv_cache.md#SlidingWindowKVCache.init_states), [`k_proj`](base_kv_cache.md#KVState.k_proj), [`v_proj`](base_kv_cache.md#KVState.v_proj), [`as_dense_kv`](paged_kv_cache.md#PagedKVCache.as_dense_kv), [`extend_step`](sliding_window_kv_cache.md#SlidingWindowKVCache.extend_step), [`extend_step`](kv_cache.md#KVCache.extend_step), [`Output`](base_kv_cache.md#BaseKVCache.Output), [`Shape`](base_kv_cache.md#BaseKVCache.Shape), [`SlidingWindowKVCache`](sliding_window_kv_cache.md#SlidingWindowKVCache)
- used by: [`BaseLayer`](../base_layer.md#BaseLayer), [`is_supported`](../flash_attention/tpu_attention.md#TPUSplashAttention.is_supported), [`KVCache`](kv_cache.md#KVCache), [`flash_attention_implementation`](../flash_attention/utils.md#flash_attention_implementation), [`is_supported`](../flash_attention/common.md#BaseSingleStepDecoding.is_supported), [`is_supported`](../flash_attention/common.md#BaseFlashAttention.is_supported), [`init_states`](kv_cache.md#KVCache.init_states), [`is_supported`](../flash_attention/common.md#BasePagedAttention.is_supported), [`init_states`](sliding_window_kv_cache.md#SlidingWindowKVCache.init_states), [`is_supported`](../flash_attention/tpu_attention.md#TPUIncrementalSplashAttention.is_supported), [`is_supported`](../flash_attention/tpu_paged_attention.md#TPUPagedAttention.is_supported), [`is_supported`](../flash_attention/tpu_attention.md#TPUFlashAttention.is_supported), [`as_dense_kv`](paged_kv_cache.md#PagedKVCache.as_dense_kv), [`is_supported`](../flash_attention/common.md#ReferenceMHA.is_supported), [`is_supported`](../flash_attention/gpu_attention.md#PallasGPUFlashAttention.is_supported), [`is_supported`](../flash_attention/gpu_paged_attention.md#GPUPagedAttention.is_supported), [`is_supported`](../flash_attention/tpu_decoding.md#TPUDecoding.is_supported), [`extend_step`](sliding_window_kv_cache.md#SlidingWindowKVCache.extend_step), [`is_supported`](../flash_attention/neuron_attention.md#NeuronFlashAttention.is_supported), [`is_supported`](../flash_attention/tpu_attention.md#LegacyTPUFlashAttention.is_supported), [`kv_cache`](../attention.md#MultiheadAttention.Config.kv_cache), [`is_supported`](../flash_attention/gpu_decoding.md#GPUDecoding.is_supported), [`extend_step`](kv_cache.md#KVCache.extend_step), [`is_supported`](../flash_attention/tpu_attention.md#TPUSplashAttentionWithAllGather.is_supported), [`is_supported`](../flash_attention/gpu_attention.md#CuDNNGPUFlashAttention.is_supported), [`Config`](sliding_window_kv_cache.md#SlidingWindowKVCache.Config), [`SlidingWindowKVCache`](sliding_window_kv_cache.md#SlidingWindowKVCache)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/kv_cache/base_kv_cache.py:35`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L35)
- doc: Configures BaseKVCache.
- signature: `class Config(BaseLayer.Config):`
- members:
  - `cache_dtype` — [`L42`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L42)
  - `kv_partition_spec` — [`L44`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L44)
- uses (calls/refs, reference-scoped): [`BaseLayer`](../base_layer.md#BaseLayer), [`config_class`](../config.md#config_class), [`Config`](../base_layer.md#BaseLayer.Config), [`PartitionSpecType`](../utils.md#PartitionSpecType)
- used by: [`enable_sliding_window_attention`](sliding_window_kv_cache.md#enable_sliding_window_attention), [`set_attention_partition_specs`](../attention.md#set_attention_partition_specs), [`kv_cache`](../attention.md#MultiheadAttention.Config.kv_cache), [`Config`](sliding_window_kv_cache.md#SlidingWindowKVCache.Config)

### `KVState`  ·  implements/extends NamedTuple
- def: [`axlearn/common/kv_cache/base_kv_cache.py:15`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L15) — documented in [axlearn-common-kv_cache-base_kv_cache](../../../../concepts/axlearn-common-kv_cache-base_kv_cache.md)
- doc: Represents key/value projections.
- signature: `class KVState(NamedTuple):`
- members:
  - `k_proj` — [`L25`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L25) — documented in [axlearn-common-kv_cache-base_kv_cache](../../../../concepts/axlearn-common-kv_cache-base_kv_cache.md)
  - `key_positions` — [`L27`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L27)
  - `page_indices` — [`L28`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L28)
  - `v_proj` — [`L26`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L26) — documented in [axlearn-common-kv_cache-base_kv_cache](../../../../concepts/axlearn-common-kv_cache-base_kv_cache.md)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Output`](base_kv_cache.md#BaseKVCache.Output)
- used by: [`_forward_for_mode`](../attention.md#MultiheadAttention._forward_for_mode), [`_compute_attention`](../flash_attention/layer.md#FlashAttention._compute_attention), [`_forward_for_mode`](../rattention/rattention.md#RAttention._forward_for_mode), [`_forward_for_mode`](../attention.md#_TransformerRepeat._forward_for_mode), `_test_extend_step`, `test_update_layer_kwargs`, `test_sigmoid_compute_attention`, `test_stacked_transformer_with_non_uniform_layers`, [`_forward_for_mode`](../attention.md#TransformerAttentionLayer._forward_for_mode), [`_forward_for_mode`](../attention.md#TransformerLayer._forward_for_mode), `test_gqa_against_mha`, [`_compute_attention`](../attention.md#MultiheadAttention._compute_attention), `test_repeated_layer_with_custom_carry`, `test_passthrough_update_layer_kwargs`, [`_compute_attention`](../attention.md#SigmoidAttention._compute_attention), `test_prescaled_kv_share`, `test_qlinear`, [`extend_step`](../attention.md#MultiheadAttention.extend_step), [`forward`](../attention.md#QLinear.forward), [`extend_step`](../rattention/rattention.md#RAttention.extend_step), [`as_dense_kv`](paged_kv_cache.md#PagedKVCache.as_dense_kv), [`extend_step`](../attention.md#BaseTransformerLayer.extend_step), [`extend_step`](../attention.md#TransformerAttentionLayer.extend_step), `forward`, [`forward`](../attention.md#BaseTransformerLayer.forward), [`forward`](../attention.md#BaseQKVLinear.forward), [`forward`](../attention.md#FusedQKVLinear.forward), [`forward`](../attention.md#MultiheadAttention.forward), [`forward`](../attention.md#RoFormerQKVLinear.forward), [`forward`](../attention.md#TransformerAttentionLayer.forward), [`as_dense_kv`](base_kv_cache.md#BaseKVCache.as_dense_kv), `forward`, [`self_attention_kv_state`](../attention.md#BaseTransformerLayer.Output.self_attention_kv_state), [`_update_layer_kwargs`](../attention.md#StackedTransformerLayer._update_layer_kwargs), [`Output`](base_kv_cache.md#BaseKVCache.Output), `_update_layer_kwargs`, [`kv_state`](../attention.md#MultiheadAttention.Output.kv_state), [`kv_state`](../attention.md#TransformerAttentionLayer.Output.kv_state)

### `Output`  ·  implements/extends KVState
- def: [`axlearn/common/kv_cache/base_kv_cache.py:46`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L46)
- signature: `class Output(KVState):`
- uses (calls/refs, reference-scoped): [`KVState`](base_kv_cache.md#KVState)
- used by: [`KVState`](base_kv_cache.md#KVState), [`extend_step`](sliding_window_kv_cache.md#SlidingWindowKVCache.extend_step), [`extend_step`](kv_cache.md#KVCache.extend_step), [`extend_step`](base_kv_cache.md#BaseKVCache.extend_step), [`extend_step`](paged_kv_cache.md#PagedKVCache.extend_step), [`prepare_proj`](sliding_window_kv_cache.md#SlidingWindowKVCache.prepare_proj)

### `Shape`  ·  implements/extends NamedTuple
- def: [`axlearn/common/kv_cache/base_kv_cache.py:49`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L49)
- doc: Shape of BaseKVCache.
- signature: `class Shape(NamedTuple):`
- members:
  - `batch_size` — [`L59`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L59)
  - `kv_len` — [`L60`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L60)
  - `num_kv_heads` — [`L61`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L61)
  - `per_head_dim` — [`L62`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/base_kv_cache.py#L62)
- used by: [`init_states`](../attention.md#MultiheadAttention.init_states), [`init_states`](kv_cache.md#KVCache.init_states), [`init_states`](sliding_window_kv_cache.md#SlidingWindowKVCache.init_states), [`init_states`](../rattention/rattention.md#RAttention.init_states), [`init_states`](base_kv_cache.md#BaseKVCache.init_states), [`init_states`](paged_kv_cache.md#PagedKVCache.init_states)

