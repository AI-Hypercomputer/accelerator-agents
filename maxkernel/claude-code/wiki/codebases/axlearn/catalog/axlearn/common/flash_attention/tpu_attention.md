---
title: 'Module: axlearn/common/flash_attention/tpu_attention.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/tpu_attention.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.tpu_attention`/
symbols:
  TPUSplashAttentionWithAllGather.build: TPUSplashAttentionWithAllGather#build().
  TPUSplashAttention.is_supported: TPUSplashAttention#is_supported().
  TPUIncrementalSplashAttention.__call__: TPUIncrementalSplashAttention#__call__().
  TPUSplashAttention.__call__: TPUSplashAttention#__call__().
  TPUIncrementalSplashAttention.is_supported: TPUIncrementalSplashAttention#is_supported().
  LegacyTPUFlashAttention.__call__: LegacyTPUFlashAttention#__call__().
  _to_splash_mask: _to_splash_mask().
  TPUFlashAttention.is_supported: TPUFlashAttention#is_supported().
  _flash_attention_impl: _flash_attention_impl().
  LegacyTPUFlashAttention.is_supported: LegacyTPUFlashAttention#is_supported().
  _flash_attention_bwd_dkv: _flash_attention_bwd_dkv().
  TPUSplashAttentionWithAllGather.is_supported: TPUSplashAttentionWithAllGather#is_supported().
  _flash_attention_bwd_dq: _flash_attention_bwd_dq().
  TPUSplashAttention.get_dropout_mask: TPUSplashAttention#get_dropout_mask().
  TPUSplashAttentionWithAllGather.shard_fn: TPUSplashAttentionWithAllGather#shard_fn().
  TPUFlashAttention: TPUFlashAttention#
  TPUSplashAttention.get_block_sizes: TPUSplashAttention#get_block_sizes().
  TPUSplashAttention: TPUSplashAttention#
  MaskFnOrZero: MaskFnOrZero.
  _flash_attention_bwd: _flash_attention_bwd().
  TPUSplashAttention._use_fused: TPUSplashAttention#_use_fused.
  _flash_attention: _flash_attention().
  TPUSplashAttentionWithAllGather: TPUSplashAttentionWithAllGather#
  TPUIncrementalSplashAttention: TPUIncrementalSplashAttention#
  LegacyTPUFlashAttention: LegacyTPUFlashAttention#
  _pallas_tpu_flash_attention: _pallas_tpu_flash_attention().
  _flash_attention_fwd: _flash_attention_fwd().
  _flash_attention_bwd_dq.qo_index_map: _flash_attention_bwd_dq().qo_index_map().
  TPUSplashAttention.__init__: TPUSplashAttention#__init__().
  _flash_attention_impl.kv_index_map: _flash_attention_impl().kv_index_map().
  _flash_attention_impl.lm_index_map: _flash_attention_impl().lm_index_map().
  _flash_attention_bwd_dkv.qo_index_map: _flash_attention_bwd_dkv().qo_index_map().
  TPUSplashAttentionWithAllGather.mul_block_len: TPUSplashAttentionWithAllGather#mul_block_len().
  _flash_attention_impl.q_index_map: _flash_attention_impl().q_index_map().
  _flash_attention_impl.ab_index_map: _flash_attention_impl().ab_index_map().
  _flash_attention_impl.o_index_map: _flash_attention_impl().o_index_map().
  _flash_attention_impl.q_segment_ids_index_map: _flash_attention_impl().q_segment_ids_index_map().
  _flash_attention_impl.kv_segment_ids_index_map: _flash_attention_impl().kv_segment_ids_index_map().
  _flash_attention_bwd_dkv.kv_index_map: _flash_attention_bwd_dkv().kv_index_map().
  _flash_attention_bwd_dkv.lm_index_map: _flash_attention_bwd_dkv().lm_index_map().
  _flash_attention_bwd_dkv.ab_index_map: _flash_attention_bwd_dkv().ab_index_map().
  _flash_attention_bwd_dkv.q_segment_ids_index_map: _flash_attention_bwd_dkv().q_segment_ids_index_map().
  _flash_attention_bwd_dkv.kv_segment_ids_index_map: _flash_attention_bwd_dkv().kv_segment_ids_index_map().
  _flash_attention_bwd_dkv.dkv_index_map: _flash_attention_bwd_dkv().dkv_index_map().
  _flash_attention_bwd_dq.kv_index_map: _flash_attention_bwd_dq().kv_index_map().
  _flash_attention_bwd_dq.lm_index_map: _flash_attention_bwd_dq().lm_index_map().
  _flash_attention_bwd_dq.ab_index_map: _flash_attention_bwd_dq().ab_index_map().
  _flash_attention_bwd_dq.q_segment_ids_index_map: _flash_attention_bwd_dq().q_segment_ids_index_map().
  _flash_attention_bwd_dq.kv_segment_ids_index_map: _flash_attention_bwd_dq().kv_segment_ids_index_map().
  TPUSplashAttentionWithAllGather.vmap_fn: TPUSplashAttentionWithAllGather#vmap_fn().
  TPUIncrementalSplashAttention.call_kernel: TPUIncrementalSplashAttention#call_kernel().
---
# Module: [`axlearn/common/flash_attention/tpu_attention.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py)

## Classes
### `LegacyTPUFlashAttention`  ·  implements/extends TPUFlashAttention
- def: [`axlearn/common/flash_attention/tpu_attention.py:1403`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1403)
- doc: Wraps the legacy (deprecated) implementation of TPU attention.
- signature: `class LegacyTPUFlashAttention(TPUFlashAttention):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L1439`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1439) — See `BaseFlashAttention.__call__`.
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L1406`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1406) — See `BaseFlashAttention.is_supported`.
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`config`](../config.md#Configurable.config), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`CausalAttentionBias`](../attention_bias.md#CausalAttentionBias), [`split`](../attention_bias.md#split), [`value`](../attention_bias.md#BaseAttentionBias.value), [`has_value`](../attention_bias.md#BaseAttentionBias.has_value), [`is_supported`](tpu_attention.md#TPUFlashAttention.is_supported), [`SegmentIdAttentionBias`](../attention_bias.md#SegmentIdAttentionBias), [`_check_block_size`](common.md#BaseFlashAttention._check_block_size), [`get_segment_ids`](common.md#get_segment_ids), [`repeat_kv_heads`](common.md#repeat_kv_heads), [`TPUFlashAttention`](tpu_attention.md#TPUFlashAttention), [`_pallas_tpu_flash_attention`](tpu_attention.md#_pallas_tpu_flash_attention)
- used by: [`BACKENDS`](utils.md#BACKENDS), [`is_supported`](tpu_attention.md#TPUFlashAttention.is_supported), [`TPUFlashAttention`](tpu_attention.md#TPUFlashAttention)

### `TPUFlashAttention`  ·  implements/extends BaseFlashAttention
- def: [`axlearn/common/flash_attention/tpu_attention.py:879`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L879)
- doc: Wraps the common checks for TPU attention implementations.
- signature: `class TPUFlashAttention(BaseFlashAttention):`
- members:
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L882`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L882) — See `BaseFlashAttention.is_supported`.
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`is_supported`](tpu_attention.md#TPUSplashAttention.is_supported), [`is_supported`](common.md#BaseFlashAttention.is_supported), [`is_supported`](tpu_attention.md#TPUIncrementalSplashAttention.is_supported), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`is_supported`](tpu_attention.md#LegacyTPUFlashAttention.is_supported), [`TPUSplashAttention`](tpu_attention.md#TPUSplashAttention), [`LegacyTPUFlashAttention`](tpu_attention.md#LegacyTPUFlashAttention), [`TPUIncrementalSplashAttention`](tpu_attention.md#TPUIncrementalSplashAttention)
- used by: [`is_supported`](tpu_attention.md#TPUSplashAttention.is_supported), [`is_supported`](common.md#BaseFlashAttention.is_supported), [`is_supported`](tpu_attention.md#TPUIncrementalSplashAttention.is_supported), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`is_supported`](tpu_attention.md#LegacyTPUFlashAttention.is_supported), [`TPUSplashAttention`](tpu_attention.md#TPUSplashAttention), [`LegacyTPUFlashAttention`](tpu_attention.md#LegacyTPUFlashAttention), [`TPUIncrementalSplashAttention`](tpu_attention.md#TPUIncrementalSplashAttention)

### `TPUIncrementalSplashAttention`  ·  implements/extends TPUFlashAttention
- def: [`axlearn/common/flash_attention/tpu_attention.py:1280`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1280)
- doc: SplashAttention with runtime block-sparse masking for incremental prefill.
- signature: `class TPUIncrementalSplashAttention(TPUFlashAttention):`
- members:
  - `call_kernel(kernel, q, k, v, q_pos)` — [`L1388`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1388)
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L1290`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1290) — See `BaseFlashAttention.is_supported`.
- protocol/private: `__call__`[`L1317`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1317)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`config`](../config.md#Configurable.config), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`MaskFnAttentionBias`](../attention_bias.md#MaskFnAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`KVCache`](../kv_cache/kv_cache.md#KVCache), [`split`](../attention_bias.md#split), [`has_value`](../attention_bias.md#BaseAttentionBias.has_value), [`is_supported`](tpu_attention.md#TPUFlashAttention.is_supported), [`SegmentIdAttentionBias`](../attention_bias.md#SegmentIdAttentionBias), [`get_backend_overrides`](common.md#BaseFlashAttention.get_backend_overrides), [`TPUFlashAttention`](tpu_attention.md#TPUFlashAttention), [`FLASH_ATTN_RESIDUAL_NAME`](remat.md#FLASH_ATTN_RESIDUAL_NAME), [`maybe_pad_inputs`](common.md#maybe_pad_inputs), [`classify_blocks`](splash_attention_mask.md#classify_blocks), [`SplashAttentionKernel`](tpu_splash_attention.md#SplashAttentionKernel), [`DEFAULT_MASK_VALUE`](tpu_splash_attention.md#DEFAULT_MASK_VALUE)
- used by: [`BACKENDS`](utils.md#BACKENDS), [`is_supported`](tpu_attention.md#TPUFlashAttention.is_supported), [`TPUFlashAttention`](tpu_attention.md#TPUFlashAttention)

### `TPUSplashAttention`  ·  implements/extends TPUFlashAttention
- def: [`axlearn/common/flash_attention/tpu_attention.py:893`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L893)
- doc: Wraps SplashAttention.
- signature: `class TPUSplashAttention(TPUFlashAttention):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L959`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L959) — See `BaseFlashAttention.__call__`.
  - `get_block_sizes(self)` — [`L1025`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1025)
  - `get_dropout_mask(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L1062`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1062) — Auxiliary function to get the dropout mask for debugging purposes.
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L907`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L907) — See `BaseFlashAttention.is_supported`.
- protocol/private: `__init__`[`L903`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L903), `_use_fused`[`L905`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L905)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`config`](../config.md#Configurable.config), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`MaskFnAttentionBias`](../attention_bias.md#MaskFnAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`split`](../attention_bias.md#split), [`SlidingWindowAttentionBias`](../attention_bias.md#SlidingWindowAttentionBias), [`_to_splash_mask`](tpu_attention.md#_to_splash_mask), [`has_value`](../attention_bias.md#BaseAttentionBias.has_value), [`is_supported`](tpu_attention.md#TPUFlashAttention.is_supported), [`SegmentIdAttentionBias`](../attention_bias.md#SegmentIdAttentionBias), [`get_backend_overrides`](common.md#BaseFlashAttention.get_backend_overrides), [`get_segment_ids`](common.md#get_segment_ids), [`is_supported`](tpu_attention.md#TPUSplashAttentionWithAllGather.is_supported), [`TPUFlashAttention`](tpu_attention.md#TPUFlashAttention), [`FLASH_ATTN_RESIDUAL_NAME`](remat.md#FLASH_ATTN_RESIDUAL_NAME), [`make_splash_mha`](tpu_splash_attention.md#make_splash_mha), [`maybe_pad_inputs`](common.md#maybe_pad_inputs), [`get_dropout_mask`](tpu_splash_attention.md#get_dropout_mask), [`__init__`](common.md#BaseFlashAttention.__init__), [`TPUSplashAttentionWithAllGather`](tpu_attention.md#TPUSplashAttentionWithAllGather)
- used by: [`build`](tpu_attention.md#TPUSplashAttentionWithAllGather.build), [`BACKENDS`](utils.md#BACKENDS), [`is_supported`](tpu_attention.md#TPUFlashAttention.is_supported), [`is_supported`](tpu_attention.md#TPUSplashAttentionWithAllGather.is_supported), [`TPUFlashAttention`](tpu_attention.md#TPUFlashAttention), [`TPUSplashAttentionWithAllGather`](tpu_attention.md#TPUSplashAttentionWithAllGather)

### `TPUSplashAttentionWithAllGather`  ·  implements/extends TPUSplashAttention
- def: [`axlearn/common/flash_attention/tpu_attention.py:1096`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1096)
- doc: Context parallel attention implementation with all-gather for sequence-sharded Q/K/V.
- signature: `class TPUSplashAttentionWithAllGather(TPUSplashAttention):`
- members:
  - `build(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L1143`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1143)
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L1123`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1123) — See `BaseFlashAttention.is_supported`.
  - `mul_block_len(seq_len)` — [`L1166`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1166)
  - `shard_fn(batch)` — [`L1198`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1198)
  - `vmap_fn(q_proj, k_proj, v_proj, kv_seg_ids, prng_key, *args)` — [`L1239`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L1239)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`config`](../config.md#Configurable.config), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`MaskFnAttentionBias`](../attention_bias.md#MaskFnAttentionBias), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`is_supported`](tpu_attention.md#TPUSplashAttention.is_supported), [`split`](../attention_bias.md#split), [`_to_splash_mask`](tpu_attention.md#_to_splash_mask), [`has_value`](../attention_bias.md#BaseAttentionBias.has_value), [`SegmentIdAttentionBias`](../attention_bias.md#SegmentIdAttentionBias), `env`, [`get_backend_overrides`](common.md#BaseFlashAttention.get_backend_overrides), [`manual_sharding_spec`](tpu_splash_attention.md#SplashAttentionKernel.manual_sharding_spec), [`FLASH_ATTN_RESIDUAL_NAME`](remat.md#FLASH_ATTN_RESIDUAL_NAME), [`tree_flatten`](tpu_splash_attention.md#SplashAttentionKernel.tree_flatten), [`get_block_sizes`](tpu_attention.md#TPUSplashAttention.get_block_sizes), [`FlashAttentionWithShardMapSpecs`](attention_types.md#FlashAttentionWithShardMapSpecs), [`make_splash_mha`](tpu_splash_attention.md#make_splash_mha), [`maybe_pad_inputs`](common.md#maybe_pad_inputs), [`TPUSplashAttention`](tpu_attention.md#TPUSplashAttention), [`_splash_attention`](tpu_splash_attention.md#_splash_attention), [`kwargs`](tpu_splash_attention.md#SplashAttentionKernel.kwargs), [`fn`](attention_types.md#FlashAttentionWithShardMapSpecs.fn), [`additional_in_specs`](attention_types.md#FlashAttentionWithShardMapSpecs.additional_in_specs), [`additional_kwargs`](attention_types.md#FlashAttentionWithShardMapSpecs.additional_kwargs)
- used by: [`is_supported`](tpu_attention.md#TPUSplashAttention.is_supported), [`BACKENDS`](utils.md#BACKENDS), [`build`](common.md#BaseFlashAttention.build), [`TPUSplashAttention`](tpu_attention.md#TPUSplashAttention)

## Functions
- `_flash_attention(q, k, v, ab, segment_ids, causal, softmax_scale, block_sizes, debug, interpret)` — [`L159`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L159)
- `_flash_attention_bwd(causal: bool, softmax_scale: float, block_sizes: LegacyBlockSizes, debug: bool, interpret: bool, residuals, do)` — [`L222`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L222) — VJP rule for FlashAttention.
- `_flash_attention_bwd_dkv(q, k, v, ab, segment_ids, l, m, do, di, *, block_q_major: Optional[int], block_q: Optional[int], block_k_major: Optional[int], block_k: Optional[int], softmax_scale: float, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE, debug: bool = False, interpret: bool = False)` — [`L487`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L487)
- `_flash_attention_bwd_dq(q, k, v, ab, segment_ids, l, m, do, di, *, block_q_major: Optional[int], block_k_major: Optional[int], block_k: Optional[int], softmax_scale: float, causal: bool, mask_value: float, debug: bool, interpret: bool)` — [`L688`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L688)
- `_flash_attention_fwd(q, k, v, ab, segment_ids, causal, softmax_scale, block_sizes, debug, interpret)` — [`L191`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L191)
- `_flash_attention_impl(q, k, v, ab, segment_ids, save_residuals, causal, softmax_scale, block_b, block_q, block_k_major, block_k, debug, interpret)` — [`L290`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L290)
- `_pallas_tpu_flash_attention(q, k, v, ab=None, segment_ids=None, *, causal: bool = False, softmax_scale: float = 1, block_sizes: Optional[LegacyBlockSizes] = None, debug: bool = False, interpret: bool = False)` — [`L110`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L110)
- `_to_splash_mask(mask: MaskFnOrZero, *, mask_shape: tuple[int, int], q_seq_shards: int = 1, unpadded_k_len: Optional[int] = None)` — [`L64`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L64) — Converts a mask to a splash mask.
- `ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L337`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L337)
- `ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index)` — [`L569`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L569)
- `ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L763`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L763)
- `dkv_index_map(batch_index, head_index, kv_seq_index, _)` — [`L641`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L641)
- `kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L324`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L324)
- `kv_index_map(batch_index, head_index, kv_seq_index, _)` — [`L549`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L549)
- `kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L733`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L733)
- `kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L413`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L413)
- `kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _)` — [`L597`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L597)
- `kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L783`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L783)
- `lm_index_map(batch_index, head_index, q_seq_index, _)` — [`L361`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L361)
- `lm_index_map(batch_index, head_index, _, q_seq_index)` — [`L557`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L557)
- `lm_index_map(batch_index, head_index, q_seq_index, _)` — [`L751`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L751)
- `o_index_map(batch_index, head_index, q_seq_index, _)` — [`L358`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L358)
- `q_index_map(batch_index, head_index, q_seq_index, _)` — [`L321`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L321)
- `q_segment_ids_index_map(batch_index, head_index, q_seq_index, _)` — [`L409`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L409)
- `q_segment_ids_index_map(batch_index, head_index, kv_seq_index, q_seq_index)` — [`L585`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L585)
- `q_segment_ids_index_map(batch_index, head_index, q_seq_index, _)` — [`L779`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L779)
- `qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index)` — [`L530`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L530)
- `qo_index_map(batch_index, head_index, q_seq_index, _)` — [`L727`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L727)

## Module values
- `MaskFnOrZero` — [`L61`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py#L61)

