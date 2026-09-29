---
title: 'Module: python/sgl_jax/srt/mem_cache/recurrent_state_pool.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/recurrent_state_pool.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.recurrent_state_pool`/
symbols:
  RecurrentStatePool.tree_unflatten: RecurrentStatePool#tree_unflatten().
  RecurrentStatePool.tree_flatten: RecurrentStatePool#tree_flatten().
  RecurrentStatePool._create_buffers: RecurrentStatePool#_create_buffers().
  RecurrentStatePool.copy_slots: RecurrentStatePool#copy_slots().
  recurrent_state_dtype: recurrent_state_dtype().
  RecurrentStatePool.replace_buffer: RecurrentStatePool#replace_buffer().
  RecurrentStatePool.conv_buffers: RecurrentStatePool#conv_buffers.
  RecurrentStatePool.get_linear_recurrent_layer_cache: RecurrentStatePool#get_linear_recurrent_layer_cache().
  LinearRecurrentStateParams.dtype: LinearRecurrentStateParams#dtype.
  RecurrentStatePool.num_linear_recurrent_layers: RecurrentStatePool#num_linear_recurrent_layers.
  RecurrentStatePool.__init__: RecurrentStatePool#__init__().
  RecurrentStatePool.clear: RecurrentStatePool#clear().
  RecurrentStatePool.linear_recurrent_layer_ids: RecurrentStatePool#linear_recurrent_layer_ids.
  RecurrentStatePool.recurrent_buffers: RecurrentStatePool#recurrent_buffers.
  LinearRecurrentStateParams: LinearRecurrentStateParams#
  RecurrentStatePool: RecurrentStatePool#
  RecurrentStatePool.layers_mapping: RecurrentStatePool#layers_mapping.
  RecurrentStatePool.data_partition_axis: RecurrentStatePool#data_partition_axis.
  LinearRecurrentStateParams.layers: LinearRecurrentStateParams#layers.
  LinearRecurrentStateParams.num_heads: LinearRecurrentStateParams#num_heads.
  LinearRecurrentStateParams.head_dim: LinearRecurrentStateParams#head_dim.
  LinearRecurrentStateParams.conv_kernel_size: LinearRecurrentStateParams#conv_kernel_size.
  RecurrentStatePool.mesh: RecurrentStatePool#mesh.
  _resolve_dtype: _resolve_dtype().
  RecurrentStateDType.conv: RecurrentStateDType#conv.
  RecurrentStateDType.temporal: RecurrentStateDType#temporal.
  RecurrentStatePool.head_dim: RecurrentStatePool#head_dim.
  RecurrentStatePool.proj_size: RecurrentStatePool#proj_size.
  RecurrentStatePool.total_slots: RecurrentStatePool#total_slots.
  RecurrentStatePool.recurrent_partition_axis: RecurrentStatePool#recurrent_partition_axis.
  RecurrentStatePool.conv_partition_axis: RecurrentStatePool#conv_partition_axis.
  RecurrentStatePool.recurrent_sharding: RecurrentStatePool#recurrent_sharding.
  RecurrentStatePool.conv_sharding: RecurrentStatePool#conv_sharding.
  RecurrentStateDType: RecurrentStateDType#
  LinearRecurrentStateParams.num_k_heads: LinearRecurrentStateParams#num_k_heads.
  LinearRecurrentStateParams.head_k_dim: LinearRecurrentStateParams#head_k_dim.
  RecurrentStatePool.temporal_dtype: RecurrentStatePool#temporal_dtype.
  RecurrentStatePool.conv_dtype: RecurrentStatePool#conv_dtype.
  RecurrentStatePool.num_heads: RecurrentStatePool#num_heads.
  RecurrentStatePool.conv_kernel_size: RecurrentStatePool#conv_kernel_size.
  RecurrentStatePool.size: RecurrentStatePool#size.
  RecurrentStatePool.dp_size: RecurrentStatePool#dp_size.
  RecurrentStatePool.num_k_heads: RecurrentStatePool#num_k_heads.
  RecurrentStatePool.head_k_dim: RecurrentStatePool#head_k_dim.
  _DTYPE_MAP: _DTYPE_MAP.
  RecurrentStatePool.slots_per_rank: RecurrentStatePool#slots_per_rank.
  RecurrentStatePool._temporal: RecurrentStatePool#_temporal().
  RecurrentStatePool._conv: RecurrentStatePool#_conv().
---
# Module: [`python/sgl_jax/srt/mem_cache/recurrent_state_pool.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py)

## Classes
### `LinearRecurrentStateParams`
- def: [`python/sgl_jax/srt/mem_cache/recurrent_state_pool.py:33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L33)
- signature: `class LinearRecurrentStateParams:`
- members:
  - `conv_kernel_size` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L37)
  - `dtype` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L38)
  - `head_dim` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L36)
  - `head_k_dim` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L44)
  - `layers` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L34)
  - `num_heads` — [`L35`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L35)
  - `num_k_heads` — [`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L43)
- uses (calls/refs, reference-scoped): [`RecurrentStateDType`](recurrent_state_pool.md#RecurrentStateDType)
- used by: [`linear_state_params`](../configs/qwen3_5.md#_Qwen3_5TextConfig.linear_state_params), [`_build_hybrid_pools`](../model_executor/model_runner_kv_cache_mixin.md#_build_hybrid_pools), [`_per_req_state_bytes_from_config`](../model_executor/model_runner_kv_cache_mixin.md#_per_req_state_bytes_from_config), [`linear_state_params`](../configs/bailing_hybrid.md#BailingHybridConfig.linear_state_params), [`_linear_state_params_from_config`](../model_executor/model_runner_kv_cache_mixin.md#_linear_state_params_from_config)

### `RecurrentStateDType`
- def: [`python/sgl_jax/srt/mem_cache/recurrent_state_pool.py:27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L27)
- signature: `class RecurrentStateDType:`
- members:
  - `conv` — [`L28`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L28)
  - `temporal` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L29)
- used by: [`_build_hybrid_pools`](../model_executor/model_runner_kv_cache_mixin.md#_build_hybrid_pools), [`_per_req_state_bytes_from_config`](../model_executor/model_runner_kv_cache_mixin.md#_per_req_state_bytes_from_config), [`recurrent_state_dtype`](recurrent_state_pool.md#recurrent_state_dtype), [`dtype`](recurrent_state_pool.md#LinearRecurrentStateParams.dtype), [`__init__`](recurrent_state_pool.md#RecurrentStatePool.__init__)

### `RecurrentStatePool`
- def: [`python/sgl_jax/srt/mem_cache/recurrent_state_pool.py:55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L55)
- signature: `class RecurrentStatePool:`
- members:
  - `__init__(self, linear_recurrent_layer_ids: list[int], size: int, num_heads: int, head_dim: int, conv_kernel_size: int, mesh: Mesh, dp_size: int = 1, recurrent_partition_axis: str = "tensor", conv_partition_axis: str = "tensor", data_partition_axis: str = "data", temporal_dtype=None, conv_dtype=None, num_k_heads: int | None = None, head_k_dim: int | None = None)` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L57) — `size` is the **global** number of valid slots across all DP ranks
  - `clear(self)` — [`L206`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L206)
  - `copy_slots(self, src_indices, dst_indices)` — [`L212`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L212) — Clone src->dst slots across all layers; rows with src==0 keep dst.
  - `get_linear_recurrent_layer_cache(self, layer_id: int)` — [`L176`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L176)
  - `replace_buffer(self, buffers)` — [`L185`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L185)
  - `tree_flatten(self)` — [`L263`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L263)
  - `tree_unflatten(cls, aux_data, children)` — [`L287`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L287)
  - `conv_buffers` — [`L147`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L147)
  - `conv_dtype` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L85)
  - `conv_kernel_size` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L113)
  - `conv_partition_axis` — [`L124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L124)
  - `conv_sharding` — [`L145`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L145)
  - `data_partition_axis` — [`L125`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L125)
  - `dp_size` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L107)
  - `head_dim` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L110)
  - `head_k_dim` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L112)
  - `layers_mapping` — [`L97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L97)
  - `linear_recurrent_layer_ids` — [`L96`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L96)
  - `mesh` — [`L122`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L122)
  - `num_heads` — [`L109`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L109)
  - `num_k_heads` — [`L111`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L111)
  - `num_linear_recurrent_layers` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L100)
  - `proj_size` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L117)
  - `recurrent_buffers` — [`L147`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L147)
  - `recurrent_partition_axis` — [`L123`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L123)
  - `recurrent_sharding` — [`L142`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L142)
  - `size` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L106)
  - `slots_per_rank` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L108)
  - `temporal_dtype` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L84)
  - `total_slots` — [`L120`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L120)
- protocol/private: `_conv`[`L225`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L225), `_create_buffers`[`L149`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L149), `_temporal`[`L218`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L218)
- uses (calls/refs, reference-scoped): [`recurrent_state_dtype`](recurrent_state_pool.md#recurrent_state_dtype), [`conv`](recurrent_state_pool.md#RecurrentStateDType.conv), [`temporal`](recurrent_state_pool.md#RecurrentStateDType.temporal)
- used by: [`__call__`](../layers/attention/linear/kda_backend.md#KDAAttnBackend.__call__), [`__call__`](../layers/attention/linear/gdn_backend.md#GDNAttnBackend.__call__), [`_build_hybrid_pools`](../model_executor/model_runner_kv_cache_mixin.md#_build_hybrid_pools)

## Functions
- `_resolve_dtype(env_var: str, default)` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L21)
- `recurrent_state_dtype()` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L47)

## Module values
- `_DTYPE_MAP` — [`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/recurrent_state_pool.py#L14)

