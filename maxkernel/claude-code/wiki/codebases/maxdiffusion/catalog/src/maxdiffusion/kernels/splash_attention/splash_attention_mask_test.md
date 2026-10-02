---
title: 'Module: src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.kernels.splash_attention.splash_attention_mask_test`/
symbols:
  SplashAttentionMaskInfoTest._assert_mask_info_match: SplashAttentionMaskInfoTest#_assert_mask_info_match().
  SplashAttentionMaskInfoTest.test_causal_two_q_shards_two_kv_shards: SplashAttentionMaskInfoTest#test_causal_two_q_shards_two_kv_shards().
  SplashAttentionMaskInfoTest.test_no_partial_mask_blocks: SplashAttentionMaskInfoTest#test_no_partial_mask_blocks().
  SplashAttentionMaskInfoTest.test_rectangular_wide_causal_mask: SplashAttentionMaskInfoTest#test_rectangular_wide_causal_mask().
  SplashAttentionMaskInfoTest.test_rectangular_tall_causal_mask: SplashAttentionMaskInfoTest#test_rectangular_tall_causal_mask().
  SplashAttentionMaskInfoTest.test_local_mask: SplashAttentionMaskInfoTest#test_local_mask().
  SplashAttentionMaskInfoTest.test_local_mask_narrow: SplashAttentionMaskInfoTest#test_local_mask_narrow().
  SplashAttentionMaskInfoTest.test_two_qseq_shards_causal_local_stacked: SplashAttentionMaskInfoTest#test_two_qseq_shards_causal_local_stacked().
  SplashAttentionMaskInfoTest.test_huge_mask: SplashAttentionMaskInfoTest#test_huge_mask().
  SplashAttentionMaskInfoTest.test_huge_mask2: SplashAttentionMaskInfoTest#test_huge_mask2().
  SplashAttentionMaskTest._compare_masks: SplashAttentionMaskTest#_compare_masks().
  SplashAttentionMaskInfoTest.test_full_mask: SplashAttentionMaskInfoTest#test_full_mask().
  SplashAttentionMaskInfoTest.test_two_shards_local_wide_local_narrow_stacked: SplashAttentionMaskInfoTest#test_two_shards_local_wide_local_narrow_stacked().
  SplashAttentionMaskInfoTest._process_mask: SplashAttentionMaskInfoTest#_process_mask().
  SplashAttentionMaskInfoTest.test_dynamic_mask: SplashAttentionMaskInfoTest#test_dynamic_mask().
  SplashAttentionMaskTest.test_causal_mask: SplashAttentionMaskTest#test_causal_mask().
  SplashAttentionMaskTest.test_local_attention_mask: SplashAttentionMaskTest#test_local_attention_mask().
  SplashAttentionMaskTest.test_local_attention_mask_wide_rectangle: SplashAttentionMaskTest#test_local_attention_mask_wide_rectangle().
  SplashAttentionMaskTest.test_local_attention_mask_tall_rectangle: SplashAttentionMaskTest#test_local_attention_mask_tall_rectangle().
  SplashAttentionMaskTest.test_lazy_causal_mask_chunking: SplashAttentionMaskTest#test_lazy_causal_mask_chunking().
  SplashAttentionMaskTest.test_lazy_local_mask_chunking: SplashAttentionMaskTest#test_lazy_local_mask_chunking().
  SplashAttentionMaskTest.test_chunked_causal_mask: SplashAttentionMaskTest#test_chunked_causal_mask().
  SplashAttentionMaskTest.test_lazy_chunked_causal_mask_chunking: SplashAttentionMaskTest#test_lazy_chunked_causal_mask_chunking().
  SplashAttentionMaskTest.test_lazy_mask_or: SplashAttentionMaskTest#test_lazy_mask_or().
  SplashAttentionMaskTest.test_lazy_mask_and: SplashAttentionMaskTest#test_lazy_mask_and().
  SplashAttentionMaskInfoTest._check_presence: SplashAttentionMaskInfoTest#_check_presence().
  _make_lazy_local_attention_mask: _make_lazy_local_attention_mask().
  _make_local_attention_mask: _make_local_attention_mask().
  SplashAttentionMaskTest.test_chunked_causal_mask_invalid_chunk_size: SplashAttentionMaskTest#test_chunked_causal_mask_invalid_chunk_size().
  SplashAttentionMaskTest.test_using_logical_operators_raises_exception: SplashAttentionMaskTest#test_using_logical_operators_raises_exception().
  SplashAttentionMaskTest.test_lazy_full_mask: SplashAttentionMaskTest#test_lazy_full_mask().
  SplashAttentionMaskInfoTest.test_process_invalid_mask: SplashAttentionMaskInfoTest#test_process_invalid_mask().
  _make_causal_mask: _make_causal_mask().
  _make_chunked_causal_mask: _make_chunked_causal_mask().
  _make_lazy_causal_mask: _make_lazy_causal_mask().
  _make_lazy_chunked_causal_mask: _make_lazy_chunked_causal_mask().
  SplashAttentionMaskTest: SplashAttentionMaskTest#
  SplashAttentionMaskTest.setUp: SplashAttentionMaskTest#setUp().
  SplashAttentionMaskTest.test_chunked_causal_mask_minimal_equality_hash: SplashAttentionMaskTest#test_chunked_causal_mask_minimal_equality_hash().
  SplashAttentionMaskInfoTest: SplashAttentionMaskInfoTest#
  SplashAttentionMaskInfoTest.test_find_bounds: SplashAttentionMaskInfoTest#test_find_bounds().
---
# Module: [`src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py)

## Classes
### `SplashAttentionMaskInfoTest`  ·  implements/extends SplashAttentionTestCase
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py:657`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L657)
- doc: Check the construction of MaskInfo from Mask.
- signature: `class SplashAttentionMaskInfoTest(test_utils.SplashAttentionTestCase):`
- members:
  - `test_causal_two_q_shards_two_kv_shards(self, return_dynamic_grid)` — [`L1394`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L1394)
  - `test_dynamic_mask(self)` — [`L1517`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L1517)
  - `test_find_bounds(self)` — [`L1579`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L1579)
  - `test_full_mask(self, is_lazy_mask: bool)` — [`L722`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L722)
  - `test_huge_mask(self)` — [`L1466`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L1466)
  - `test_huge_mask2(self)` — [`L1485`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L1485)
  - `test_local_mask(self, is_lazy_mask: bool)` — [`L981`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L981)
  - `test_local_mask_narrow(self, is_lazy_mask: bool)` — [`L1056`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L1056)
  - `test_no_partial_mask_blocks(self)` — [`L747`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L747)
  - `test_process_invalid_mask(self)` — [`L1504`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L1504) — Masks with of an all-0 row causes undefined softmax, reject them.
  - `test_rectangular_tall_causal_mask(self, is_lazy_mask: bool)` — [`L865`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L865)
  - `test_rectangular_wide_causal_mask(self, is_lazy_mask: bool, return_dynamic_grid: bool)` — [`L782`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L782)
  - `test_two_qseq_shards_causal_local_stacked(self)` — [`L1114`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L1114)
  - `test_two_shards_local_wide_local_narrow_stacked(self, q_seq_shards, kv_seq_shards)` — [`L1246`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L1246)
- protocol/private: `_assert_mask_info_match`[`L660`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L660), `_check_presence`[`L661`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L661), `_process_mask`[`L715`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L715)
- uses (calls/refs, reference-scoped): [`_assert_array_equal`](splash_attention_test_utils.md#SplashAttentionTestCase._assert_array_equal), [`MaskInfo`](splash_attention_mask_info.md#MaskInfo), [`partial_mask_blocks`](splash_attention_mask_info.md#MaskInfo.partial_mask_blocks), [`block_mask`](splash_attention_mask_info.md#MaskInfo.block_mask), [`q_sequence`](splash_attention_mask_info.md#MaskInfo.q_sequence), [`active_rows`](splash_attention_mask_info.md#MaskInfo.active_rows), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`mask_next`](splash_attention_mask_info.md#MaskInfo.mask_next), [`active_cols`](splash_attention_mask_info.md#MaskInfo.active_cols), [`num_active_blocks`](splash_attention_mask_info.md#MaskInfo.num_active_blocks), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase), [`process_mask`](splash_attention_mask_info.md#process_mask), [`make_causal_mask`](splash_attention_mask.md#make_causal_mask), [`CausalMask`](splash_attention_mask.md#CausalMask), [`LocalMask`](splash_attention_mask.md#LocalMask), [`FullMask`](splash_attention_mask.md#FullMask), [`make_local_attention_mask`](splash_attention_mask.md#make_local_attention_mask), [`_check_mask`](splash_attention_mask_info.md#_check_mask), [`process_mask_dkv`](splash_attention_mask_info.md#process_mask_dkv), [`_make_causal_mask`](splash_attention_mask_test.md#_make_causal_mask), [`find_bounds`](splash_attention_mask_info.md#find_bounds), [`process_dynamic_mask`](splash_attention_mask_info.md#process_dynamic_mask)
- used by: [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase)

### `SplashAttentionMaskTest`  ·  implements/extends SplashAttentionTestCase
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py:59`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L59)
- signature: `class SplashAttentionMaskTest(test_utils.SplashAttentionTestCase):`
- members:
  - `setUp(self)` — [`L61`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L61)
  - `test_causal_mask(self, make_causal_mask)` — [`L67`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L67)
  - `test_chunked_causal_mask(self, make_chunked_mask)` — [`L423`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L423) — Tests the chunked causal mask logic for various shapes and chunk sizes.
  - `test_chunked_causal_mask_invalid_chunk_size(self)` — [`L550`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L550) — Tests that invalid chunk_size raises ValueError.
  - `test_chunked_causal_mask_minimal_equality_hash(self)` — [`L559`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L559) — Tests for __eq__ and __hash__ of ChunkedCausalMask.
  - `test_lazy_causal_mask_chunking(self, block_size: tuple[int, int], shape: tuple[int, int])` — [`L369`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L369)
  - `test_lazy_chunked_causal_mask_chunking(self, block_size: tuple[int, int], shape: tuple[int, int], chunk_size: int)` — [`L525`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L525) — Compares lazy chunked mask evaluation against the dense version block-by-block.
  - `test_lazy_full_mask(self, shape: tuple[int, int])` — [`L625`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L625)
  - `test_lazy_local_mask_chunking(self, block_size: tuple[int, int], shape: tuple[int, int], window_size: tuple[int | None, int | None], offset: int)` — [`L408`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L408)
  - `test_lazy_mask_and(self, shape: tuple[int, int])` — [`L615`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L615)
  - `test_lazy_mask_or(self, shape: tuple[int, int])` — [`L605`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L605)
  - `test_local_attention_mask(self, make_local_attention_mask)` — [`L147`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L147)
  - `test_local_attention_mask_tall_rectangle(self, make_local_attention_mask)` — [`L289`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L289)
  - `test_local_attention_mask_wide_rectangle(self, make_local_attention_mask)` — [`L222`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L222)
  - `test_using_logical_operators_raises_exception(self)` — [`L586`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L586)
- protocol/private: `_compare_masks`[`L631`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L631)
- uses (calls/refs, reference-scoped): [`_assert_array_equal`](splash_attention_test_utils.md#SplashAttentionTestCase._assert_array_equal), [`Mask`](splash_attention_mask.md#Mask), [`shape`](splash_attention_mask.md#Mask.shape), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase), [`ChunkedCausalMask`](splash_attention_mask.md#ChunkedCausalMask), [`make_causal_mask`](splash_attention_mask.md#make_causal_mask), [`CausalMask`](splash_attention_mask.md#CausalMask), [`LocalMask`](splash_attention_mask.md#LocalMask), [`FullMask`](splash_attention_mask.md#FullMask), [`make_local_attention_mask`](splash_attention_mask.md#make_local_attention_mask), [`make_random_mask`](splash_attention_mask.md#make_random_mask), [`_make_lazy_local_attention_mask`](splash_attention_mask_test.md#_make_lazy_local_attention_mask), [`_make_local_attention_mask`](splash_attention_mask_test.md#_make_local_attention_mask), [`_make_causal_mask`](splash_attention_mask_test.md#_make_causal_mask), [`_make_chunked_causal_mask`](splash_attention_mask_test.md#_make_chunked_causal_mask), [`_make_lazy_causal_mask`](splash_attention_mask_test.md#_make_lazy_causal_mask), [`_make_lazy_chunked_causal_mask`](splash_attention_mask_test.md#_make_lazy_chunked_causal_mask), [`make_chunk_attention_mask`](splash_attention_mask.md#make_chunk_attention_mask)
- used by: [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase)

## Functions
- `_make_causal_mask(*args, **kwargs)` — [`L37`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L37)
- `_make_chunked_causal_mask(shape, chunk_size)` — [`L55`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L55)
- `_make_lazy_causal_mask(*args, **kwargs)` — [`L32`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L32)
- `_make_lazy_chunked_causal_mask(shape, chunk_size)` — [`L50`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L50)
- `_make_lazy_local_attention_mask(*args, **kwargs)` — [`L41`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L41)
- `_make_local_attention_mask(*args, **kwargs)` — [`L46`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_mask_test.py#L46)

