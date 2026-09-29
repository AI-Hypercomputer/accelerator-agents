---
title: 'Module: axlearn/common/flash_attention/common.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/common.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.common`/
symbols:
  BaseFlashAttention._log_unsupported: BaseFlashAttention#_log_unsupported().
  BaseFlashAttention.cfg: BaseFlashAttention#cfg.
  BaseSingleStepDecoding.is_supported: BaseSingleStepDecoding#is_supported().
  BaseFlashAttention.is_supported: BaseFlashAttention#is_supported().
  BasePagedAttention.is_supported: BasePagedAttention#is_supported().
  ReferenceMHA.__call__: ReferenceMHA#__call__().
  BaseFlashAttention: BaseFlashAttention#
  ReferenceMHA.is_supported: ReferenceMHA#is_supported().
  BaseFlashAttention.get_backend_overrides: BaseFlashAttention#get_backend_overrides().
  BaseFlashAttention._check_block_size: BaseFlashAttention#_check_block_size().
  get_segment_ids: get_segment_ids().
  BaseFlashAttention.build: BaseFlashAttention#build().
  BaseSingleStepDecoding: BaseSingleStepDecoding#
  BaseFlashAttention._validate_input_batch: BaseFlashAttention#_validate_input_batch().
  BasePagedAttention._validate_input_batch: BasePagedAttention#_validate_input_batch().
  BasePagedAttention._check_block_size: BasePagedAttention#_check_block_size().
  BasePagedAttention: BasePagedAttention#
  repeat_kv_heads: repeat_kv_heads().
  query_iterator_indices: query_iterator_indices().
  get_gpu_dot_precision: get_gpu_dot_precision().
  build_mask: build_mask().
  BaseFlashAttention.Config.dropout_rate: BaseFlashAttention#Config#dropout_rate.
  BaseFlashAttention.Config: BaseFlashAttention#Config#
  maybe_pad_inputs: maybe_pad_inputs().
  BasePagedAttention.Config: BasePagedAttention#Config#
  ReferenceMHA: ReferenceMHA#
  BaseFlashAttention.Config.softmax_scale: BaseFlashAttention#Config#softmax_scale.
  BaseFlashAttention.Config.gpu_block_size: BaseFlashAttention#Config#gpu_block_size.
  BaseFlashAttention.__call__: BaseFlashAttention#__call__().
  BasePagedAttention.__call__: BasePagedAttention#__call__().
  BaseFlashAttention.Config.interpret: BaseFlashAttention#Config#interpret.
  split_prng_keys_for_shard_map: split_prng_keys_for_shard_map().
  BaseFlashAttention.__init__: BaseFlashAttention#__init__().
  BaseFlashAttention.Config.tpu_block_size: BaseFlashAttention#Config#tpu_block_size.
  KVOffsetInfo.kv_block_offset: KVOffsetInfo#kv_block_offset.
  KVOffsetInfo.kv_block_offset_size: KVOffsetInfo#kv_block_offset_size.
  maybe_pad_inputs.pad_fn: maybe_pad_inputs().pad_fn().
  build_sliding_window_mask: build_sliding_window_mask().
  KVOffsetInfo: KVOffsetInfo#
  BaseFlashAttention.name: BaseFlashAttention#name().
  build_mask.worker: build_mask().worker().
  BaseFlashAttention.Config.backend_overrides: BaseFlashAttention#Config#backend_overrides.
  get_cpu_dot_precision: get_cpu_dot_precision().
  split_prng_keys_for_shard_map._axis_size: split_prng_keys_for_shard_map()._axis_size().
  BasePagedAttention.Config.sparse_ratio: BasePagedAttention#Config#sparse_ratio.
---
# Module: [`axlearn/common/flash_attention/common.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py)

## Classes
### `BaseFlashAttention`  ·  implements/extends Configurable
- def: [`axlearn/common/flash_attention/common.py:145`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L145)
- doc: Common interface of Flash attention for all backends.
- signature: `class BaseFlashAttention(Configurable):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L195`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L195) — Computes attention context.
  - `_check_block_size(self, input_batch: Nested[Tensor | BaseAttentionBias], *, block_size: int)` — [`L278`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L278) — Returns whether the attention kernel supports the given block size.
  - `_log_unsupported(self, reason: str)` — [`L179`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L179) — Logs this class is unsupported with `reason`. — documented in [axlearn-common-flash_attention-common](../../../../concepts/axlearn-common-flash_attention-common.md)
  - `_validate_input_batch(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L216`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L216) — Returns whether the input batch is valid for the flash attention call.
  - `build(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L300`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L300) — Builds sharding specifications for the flash attention operation.
  - `get_backend_overrides(self, name: str, default: Any)` — [`L172`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L172)
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L233`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L233) — Returns whether the attention kernel supports the given configuration. — documented in [axlearn-common-flash_attention-common](../../../../concepts/axlearn-common-flash_attention-common.md)
  - `name(self)` — [`L175`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L175) — Returns the class name.
  - `cfg` — [`L170`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L170) — documented in [axlearn-common-flash_attention-common](../../../../concepts/axlearn-common-flash_attention-common.md)
- protocol/private: `__init__`[`L168`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L168)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`Configurable`](../config.md#Configurable), [`build`](tpu_attention.md#TPUSplashAttentionWithAllGather.build), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`is_supported`](common.md#BaseSingleStepDecoding.is_supported), [`is_supported`](tpu_attention.md#TPUFlashAttention.is_supported), [`is_supported`](common.md#ReferenceMHA.is_supported), [`is_supported`](gpu_attention.md#PallasGPUFlashAttention.is_supported), [`validate_contains_paths`](../utils.md#validate_contains_paths), [`is_supported`](neuron_attention.md#NeuronFlashAttention.is_supported), [`BaseSingleStepDecoding`](common.md#BaseSingleStepDecoding), [`_check_block_size`](common.md#BasePagedAttention._check_block_size), [`_validate_input_batch`](common.md#BasePagedAttention._validate_input_batch), [`TPUFlashAttention`](tpu_attention.md#TPUFlashAttention), [`is_supported`](gpu_attention.md#CuDNNGPUFlashAttention.is_supported), [`__init__`](../config.md#Configurable.__init__), [`Config`](common.md#BaseFlashAttention.Config), [`FlashAttentionWithShardMapSpecs`](attention_types.md#FlashAttentionWithShardMapSpecs), [`ReferenceMHA`](common.md#ReferenceMHA), [`CuDNNGPUFlashAttention`](gpu_attention.md#CuDNNGPUFlashAttention), [`PallasGPUFlashAttention`](gpu_attention.md#PallasGPUFlashAttention), [`NeuronFlashAttention`](neuron_attention.md#NeuronFlashAttention), [`fn`](attention_types.md#FlashAttentionWithShardMapSpecs.fn), [`backend_overrides`](common.md#BaseFlashAttention.Config.backend_overrides)
- used by: [`Configurable`](../config.md#Configurable), [`__call__`](gpu_attention.md#CuDNNGPUFlashAttention.__call__), [`__call__`](tpu_paged_attention.md#TPUPagedAttention.__call__), [`__call__`](gpu_attention.md#PallasGPUFlashAttention.__call__), [`is_supported`](tpu_attention.md#TPUSplashAttention.is_supported), [`flash_attention_implementation`](utils.md#flash_attention_implementation), [`__call__`](tpu_attention.md#TPUIncrementalSplashAttention.__call__), [`__call__`](tpu_decoding.md#TPUDecoding.__call__), [`is_supported`](common.md#BaseSingleStepDecoding.is_supported), [`_kv_cache_type`](gpu_attention.md#CuDNNGPUFlashAttention._kv_cache_type), [`is_supported`](common.md#BasePagedAttention.is_supported), [`__call__`](common.md#ReferenceMHA.__call__), [`__call__`](gpu_paged_attention.md#GPUPagedAttention.__call__), [`__call__`](neuron_attention.md#NeuronFlashAttention.__call__), [`is_supported`](tpu_attention.md#TPUIncrementalSplashAttention.is_supported), [`__call__`](gpu_decoding.md#GPUDecoding.__call__), [`is_supported`](tpu_paged_attention.md#TPUPagedAttention.is_supported), [`is_supported`](tpu_attention.md#TPUFlashAttention.is_supported), [`is_supported`](common.md#ReferenceMHA.is_supported), [`is_supported`](gpu_attention.md#PallasGPUFlashAttention.is_supported), [`is_supported`](gpu_paged_attention.md#GPUPagedAttention.is_supported), [`is_supported`](tpu_decoding.md#TPUDecoding.is_supported), [`is_supported`](neuron_attention.md#NeuronFlashAttention.is_supported), [`is_supported`](tpu_attention.md#LegacyTPUFlashAttention.is_supported), [`is_supported`](gpu_decoding.md#GPUDecoding.is_supported), [`is_supported`](tpu_attention.md#TPUSplashAttentionWithAllGather.is_supported), [`BaseSingleStepDecoding`](common.md#BaseSingleStepDecoding), [`megacore_mode_heuristic`](tpu_paged_attention.md#TPUPagedAttention.megacore_mode_heuristic), [`_check_block_size`](common.md#BasePagedAttention._check_block_size), [`_validate_input_batch`](common.md#BasePagedAttention._validate_input_batch), [`TPUFlashAttention`](tpu_attention.md#TPUFlashAttention), [`is_supported`](gpu_attention.md#CuDNNGPUFlashAttention.is_supported), [`sparse_mode_heuristic`](tpu_paged_attention.md#TPUPagedAttention.sparse_mode_heuristic), [`get_block_sizes`](tpu_attention.md#TPUSplashAttention.get_block_sizes), [`ReferenceMHA`](common.md#ReferenceMHA), [`CuDNNGPUFlashAttention`](gpu_attention.md#CuDNNGPUFlashAttention), [`PallasGPUFlashAttention`](gpu_attention.md#PallasGPUFlashAttention), [`NeuronFlashAttention`](neuron_attention.md#NeuronFlashAttention), [`__init__`](tpu_attention.md#TPUSplashAttention.__init__)

### `BasePagedAttention`  ·  implements/extends BaseSingleStepDecoding
- def: [`axlearn/common/flash_attention/common.py:344`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L344)
- doc: Base class for paged attention.
- signature: `class BasePagedAttention(BaseSingleStepDecoding):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L418`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L418) — Computes attention context.
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L371`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L371) — Returns whether paged attention kernel supports the given config.
- protocol/private: `_check_block_size`[`L357`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L357), `_validate_input_batch`[`L353`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L353)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`cfg`](common.md#BaseFlashAttention.cfg), [`is_supported`](tpu_paged_attention.md#TPUPagedAttention.is_supported), [`is_supported`](gpu_paged_attention.md#GPUPagedAttention.is_supported), [`validate_contains_paths`](../utils.md#validate_contains_paths), [`PagedKVCache`](../kv_cache/paged_kv_cache.md#PagedKVCache), [`BaseSingleStepDecoding`](common.md#BaseSingleStepDecoding), [`_validate_input_batch`](common.md#BaseFlashAttention._validate_input_batch), [`dropout_rate`](common.md#BaseFlashAttention.Config.dropout_rate), [`GPUPagedAttention`](gpu_paged_attention.md#GPUPagedAttention), [`TPUPagedAttention`](tpu_paged_attention.md#TPUPagedAttention)
- used by: [`is_supported`](common.md#BaseSingleStepDecoding.is_supported), [`is_supported`](tpu_paged_attention.md#TPUPagedAttention.is_supported), [`is_supported`](common.md#ReferenceMHA.is_supported), [`is_supported`](gpu_paged_attention.md#GPUPagedAttention.is_supported), [`_check_block_size`](common.md#BaseFlashAttention._check_block_size), [`BaseSingleStepDecoding`](common.md#BaseSingleStepDecoding), [`_validate_input_batch`](common.md#BaseFlashAttention._validate_input_batch), [`Config`](tpu_paged_attention.md#TPUPagedAttention.Config), [`GPUPagedAttention`](gpu_paged_attention.md#GPUPagedAttention), [`TPUPagedAttention`](tpu_paged_attention.md#TPUPagedAttention)

### `BaseSingleStepDecoding`  ·  implements/extends BaseFlashAttention
- def: [`axlearn/common/flash_attention/common.py:323`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L323)
- doc: Wraps the common checks for single step decoding kernels.
- signature: `class BaseSingleStepDecoding(BaseFlashAttention):`
- members:
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L326`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L326) — See `BaseFlashAttention.is_supported`. — documented in [axlearn-common-flash_attention-common](../../../../concepts/axlearn-common-flash_attention-common.md)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`cfg`](common.md#BaseFlashAttention.cfg), [`KVCache`](../kv_cache/kv_cache.md#KVCache), [`is_supported`](common.md#BaseFlashAttention.is_supported), [`is_supported`](common.md#BasePagedAttention.is_supported), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`is_supported`](tpu_decoding.md#TPUDecoding.is_supported), [`is_supported`](gpu_decoding.md#GPUDecoding.is_supported), [`PagedKVCache`](../kv_cache/paged_kv_cache.md#PagedKVCache), [`BasePagedAttention`](common.md#BasePagedAttention), [`dropout_rate`](common.md#BaseFlashAttention.Config.dropout_rate), [`GPUDecoding`](gpu_decoding.md#GPUDecoding), [`TPUDecoding`](tpu_decoding.md#TPUDecoding)
- used by: [`is_supported`](common.md#BaseFlashAttention.is_supported), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`is_supported`](tpu_decoding.md#TPUDecoding.is_supported), [`is_supported`](gpu_decoding.md#GPUDecoding.is_supported), [`BasePagedAttention`](common.md#BasePagedAttention), [`Config`](common.md#BasePagedAttention.Config), [`GPUDecoding`](gpu_decoding.md#GPUDecoding), [`TPUDecoding`](tpu_decoding.md#TPUDecoding)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/flash_attention/common.py:348`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L348)
- doc: Configures Paged Attention.
- signature: `class Config(BaseSingleStepDecoding.Config):`
- members:
  - `backend_overrides` — [`L166`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L166)
  - `dropout_rate` — [`L162`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L162)
  - `gpu_block_size` — [`L165`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L165)
  - `interpret` — [`L163`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L163)
  - `softmax_scale` — [`L161`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L161)
  - `sparse_ratio` — [`L351`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L351)
  - `tpu_block_size` — [`L164`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L164)
- uses (calls/refs, reference-scoped): [`config_class`](../config.md#config_class), [`BaseSingleStepDecoding`](common.md#BaseSingleStepDecoding), [`Config`](common.md#BaseFlashAttention.Config)
- used by: [`__call__`](gpu_attention.md#CuDNNGPUFlashAttention.__call__), [`__call__`](tpu_paged_attention.md#TPUPagedAttention.__call__), [`__call__`](gpu_attention.md#PallasGPUFlashAttention.__call__), [`__call__`](tpu_decoding.md#TPUDecoding.__call__), [`is_supported`](common.md#BaseSingleStepDecoding.is_supported), [`_kv_cache_type`](gpu_attention.md#CuDNNGPUFlashAttention._kv_cache_type), [`is_supported`](common.md#BasePagedAttention.is_supported), [`__call__`](common.md#ReferenceMHA.__call__), [`__call__`](gpu_paged_attention.md#GPUPagedAttention.__call__), [`__call__`](neuron_attention.md#NeuronFlashAttention.__call__), [`__call__`](gpu_decoding.md#GPUDecoding.__call__), [`is_supported`](tpu_paged_attention.md#TPUPagedAttention.is_supported), [`is_supported`](gpu_attention.md#PallasGPUFlashAttention.is_supported), [`is_supported`](gpu_paged_attention.md#GPUPagedAttention.is_supported), [`is_supported`](tpu_decoding.md#TPUDecoding.is_supported), [`is_supported`](neuron_attention.md#NeuronFlashAttention.is_supported), [`get_backend_overrides`](common.md#BaseFlashAttention.get_backend_overrides), [`megacore_mode_heuristic`](tpu_paged_attention.md#TPUPagedAttention.megacore_mode_heuristic), [`Config`](tpu_paged_attention.md#TPUPagedAttention.Config)

### `KVOffsetInfo`  ·  implements/extends NamedTuple
- def: [`axlearn/common/flash_attention/common.py:108`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L108)
- doc: Records the block index of non-empty KV blocks.
- signature: `class KVOffsetInfo(NamedTuple):`
- members:
  - `kv_block_offset` — [`L119`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L119)
  - `kv_block_offset_size` — [`L120`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L120)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor)
- used by: [`query_iterator_indices`](common.md#query_iterator_indices)

### `ReferenceMHA`  ·  implements/extends BaseFlashAttention
- def: [`axlearn/common/flash_attention/common.py:488`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L488)
- doc: The reference implementation of attention in XLA.
- signature: `class ReferenceMHA(BaseFlashAttention):`
- members:
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L522`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L522)
- protocol/private: `__call__`[`L493`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L493)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`cfg`](common.md#BaseFlashAttention.cfg), [`is_supported`](common.md#BaseFlashAttention.is_supported), [`is_supported`](common.md#BasePagedAttention.is_supported), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`value`](../attention_bias.md#BaseAttentionBias.value), [`PagedKVCache`](../kv_cache/paged_kv_cache.md#PagedKVCache), [`BasePagedAttention`](common.md#BasePagedAttention), [`softmax_with_biases`](../attention.md#softmax_with_biases), [`reconstruct_kv`](../kv_cache/paged_kv_storage.md#reconstruct_kv), [`dropout_rate`](common.md#BaseFlashAttention.Config.dropout_rate), [`dropout`](../layers.md#dropout), [`softmax_scale`](common.md#BaseFlashAttention.Config.softmax_scale), [`compute_gqa_context`](../attention.md#compute_gqa_context), [`compute_gqa_logits`](../attention.md#compute_gqa_logits)
- used by: [`is_supported`](common.md#BaseFlashAttention.is_supported), [`bench_flash_attention`](gpu_attention_benchmark.md#bench_flash_attention), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`BACKENDS`](utils.md#BACKENDS), [`PAGED_ATTN_BACKENDS`](utils.md#PAGED_ATTN_BACKENDS)

## Functions
- `_axis_size(axis)` — [`L651`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L651)
- `build_mask(mask_fn: MaskFn, *, q_seq_len: int, kv_seq_len: int, block_q: int, block_k: int)` — [`L34`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L34) — Builds the block map where True means the block is not fully masked.
- `build_sliding_window_mask(*, q_seq_len: int, kv_seq_len: int, block_q: int, block_k: int, sliding_window_size: int)` — [`L84`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L84) — Same as build_mask(sliding_window_causal_mask(sliding_window_size), **kwargs).
- `get_cpu_dot_precision(dtype)` — [`L539`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L539) — Get the suitable DotAlgorithmPreset for the given dtype for CPU backend.
- `get_gpu_dot_precision(dtype)` — [`L555`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L555) — Get the suitable DotAlgorithmPreset for the given dtype.
- `get_segment_ids(*, query: Tensor, key: Tensor, segment_ids: SegmentIdAttentionBias)` — [`L458`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L458) — Return the segment ids Tensor from the sequence of segment ids attention
- `maybe_pad_inputs(block_size: int, query: Tensor, key: Tensor, value: Tensor, segment_id: Optional[Tensor])` — [`L573`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L573) — Pads query, key, value, and segment_id tensors to align with block_size requirements.
- `pad_fn(x, pad)` — [`L612`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L612)
- `query_iterator_indices(block_mask_map: np.ndarray, *, padding: int = 0)` — [`L123`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L123) — Builds `KVOffsetInfo` for block-sparse attention computation in the forward pass.
- `repeat_kv_heads(num_q_heads: int, key_or_value: Tensor)` — [`L476`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L476) — Repeats key or value heads dim to match the query.
- `split_prng_keys_for_shard_map(prng_key: Tensor, sharding_spec: PartitionSpec, mesh: Mesh)` — [`L634`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L634) — Pre-split PRNG keys for shard_map to ensure unique randomness per device.
- `worker()` — [`L52`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common.py#L52)

