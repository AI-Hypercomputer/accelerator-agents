---
title: 'Module: src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.kernels.splash_attention.splash_attention_mask_info`/
symbols:
  _process_mask: _process_mask().
  MaskInfo: MaskInfo#
  MaskInfo.partial_mask_blocks: MaskInfo#partial_mask_blocks.
  MaskInfo.block_mask: MaskInfo#block_mask.
  MaskInfo.q_sequence: MaskInfo#q_sequence.
  MaskInfo.active_rows: MaskInfo#active_rows.
  _process_dynamic_mask: _process_dynamic_mask().
  MaskInfo.mask_next: MaskInfo#mask_next.
  MaskInfo.active_cols: MaskInfo#active_cols.
  MaskInfo.num_active_blocks: MaskInfo#num_active_blocks.
  process_mask: process_mask.
  _check_mask: _check_mask().
  process_mask_dkv: process_mask_dkv.
  _HashableNDArray.__init__: _HashableNDArray#__init__().
  _HashableNDArray.__eq__: _HashableNDArray#__eq__().
  find_bounds: find_bounds().
  _downcast_to_small_type: _downcast_to_small_type().
  _HashableNDArray: _HashableNDArray#
  _process_mask.pad_slice: _process_mask().pad_slice().
  process_dynamic_mask: process_dynamic_mask.
  _HashableNDArray.array: _HashableNDArray#array.
  _HashableNDArray.__hash__: _HashableNDArray#__hash__().
  process_dynamic_mask_dkv: process_dynamic_mask_dkv.
  lax: lax.
  MaskCallable: MaskCallable.
  _generate_shard_metadata: _generate_shard_metadata().
  _process_dynamic_mask._downcast: _process_dynamic_mask()._downcast().
  _HashableNDArray.__slots__: _HashableNDArray#__slots__.
---
# Module: [`src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py)

## Classes
### `MaskInfo`  ·  implements/extends NamedTuple
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py:51`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L51) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- doc: Contains runtime masking information for the Splash attention kernel.
- signature: `class MaskInfo(NamedTuple):`
- members:
  - `active_cols` — [`L84`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L84) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
  - `active_rows` — [`L83`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L83) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
  - `block_mask` — [`L85`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L85) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
  - `mask_next` — [`L82`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L82) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
  - `num_active_blocks` — [`L86`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L86)
  - `partial_mask_blocks` — [`L87`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L87) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_mask_info](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_mask_info.md)
  - `q_sequence` — [`L88`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L88) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- used by: [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward), [`_splash_attention_forward_ring_raw`](splash_attention_kernel.md#_splash_attention_forward_ring_raw), [`_splash_attention_bwd_dkv`](splash_attention_kernel.md#_splash_attention_bwd_dkv), [`_assert_mask_info_match`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest._assert_mask_info_match), [`make_ring_attention`](ring_attention_kernel.md#make_ring_attention), [`_make_dynamic_splash_attention`](splash_attention_kernel.md#_make_dynamic_splash_attention), [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`_process_mask`](splash_attention_mask_info.md#_process_mask), [`manual_sharding_spec`](ring_attention_kernel.md#RingSplashAttentionKernel.manual_sharding_spec), [`manual_sharding_spec`](splash_attention_kernel.md#SplashAttentionKernel.manual_sharding_spec), [`test_causal_two_q_shards_two_kv_shards`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_causal_two_q_shards_two_kv_shards), [`_process_dynamic_mask`](splash_attention_mask_info.md#_process_dynamic_mask), [`test_no_partial_mask_blocks`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_no_partial_mask_blocks), [`_dynamic_slice_mask_info`](ring_attention_kernel.md#_dynamic_slice_mask_info), [`MaskInfo`](splash_attention_kernel.md#MaskInfo), [`test_rectangular_wide_causal_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_rectangular_wide_causal_mask), [`MaskInfo`](ring_attention_kernel.md#MaskInfo), [`test_huge_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_huge_mask), [`test_huge_mask2`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_huge_mask2), [`test_local_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_local_mask), [`test_local_mask_narrow`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_local_mask_narrow), [`test_rectangular_tall_causal_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_rectangular_tall_causal_mask), [`test_two_qseq_shards_causal_local_stacked`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_two_qseq_shards_causal_local_stacked), [`test_full_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_full_mask), [`test_two_shards_local_wide_local_narrow_stacked`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_two_shards_local_wide_local_narrow_stacked), [`test_dynamic_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_dynamic_mask), [`MaskInfo`](base.md#MaskInfo), [`tree_unflatten`](ring_attention_kernel.md#RingSplashAttentionKernel.tree_unflatten)

### `_HashableNDArray`
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py:160`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L160)
- doc: Helper to make a numpy array hashable: can be added associative containers.
- signature: `class _HashableNDArray:`
- members:
  - `array` — [`L168`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L168)
- protocol/private: `__eq__`[`L177`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L177), `__hash__`[`L174`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L174), `__init__`[`L170`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L170), `__slots__`[`L167`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L167)
- used by: [`_process_mask`](splash_attention_mask_info.md#_process_mask)

## Functions
- `_check_mask(mask: mask_lib.Mask)` — [`L127`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L127) — Check that the given mask is valid.
- `_downcast(array: jax.Array, max_value: int)` — [`L318`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L318)
- `_downcast_to_small_type(array: np.ndarray)` — [`L91`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L91) — Downcast numpy array. — documented in [maxdiffusion-kernels-splash_attention-splash_attention_mask_info](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_mask_info.md)
- `_generate_shard_metadata(block_mask: np.ndarray, partial_blocks: np.ndarray, is_dkv: bool, return_dynamic_grid: bool)` — [`L183`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L183)
- `_process_dynamic_mask(mask: jax.Array, block_shape: tuple[int, int], is_dkv: bool, *, downcast_smem_data: bool = True, partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8)` — [`L231`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L231) — Process a dynamic mask to compute it's local sparsity data. — documented in [maxdiffusion-kernels-splash_attention-splash_attention_mask_info](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_mask_info.md)
- `_process_mask(mask: mask_lib.Mask, block_shape: tuple[int, int], is_dkv: bool, *, downcast_smem_data: bool = True, partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8, q_seq_shards: int = 1, kv_seq_shards: int = 1, return_dynamic_grid: bool = True)` — [`L352`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L352) — Transform a dense mask into a sparse representation. — documented in [maxdiffusion-kernels-splash_attention-splash_attention_mask](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_mask.md)
- `find_bounds(arr: jax.Array | np.ndarray)` — [`L33`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L33)
- `pad_slice(arr)` — [`L520`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L520)

## Module values
- `MaskCallable` — [`L30`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L30)
- `lax` — [`L29`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L29)
- `process_dynamic_mask` — [`L570`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L570)
- `process_dynamic_mask_dkv` — [`L571`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L571)
- `process_mask` — [`L567`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L567)
- `process_mask_dkv` — [`L568`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_info.py#L568)

