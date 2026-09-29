---
title: 'Module: tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_mask_test`/
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
# Module: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py)

## Classes
### `SplashAttentionMaskInfoTest`  ·  implements/extends SplashAttentionTestCase
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py:684`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L684)
- doc: Check the construction of MaskInfo from Mask.
- signature: `class SplashAttentionMaskInfoTest(test_utils.SplashAttentionTestCase):`
- members:
  - `test_causal_two_q_shards_two_kv_shards(self, return_dynamic_grid)` — [`L1537`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L1537)
  - `test_dynamic_mask(self)` — [`L1675`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L1675)
  - `test_find_bounds(self)` — [`L1737`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L1737)
  - `test_full_mask(self, is_lazy_mask: bool)` — [`L753`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L753)
  - `test_huge_mask(self)` — [`L1618`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L1618)
  - `test_huge_mask2(self)` — [`L1641`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L1641)
  - `test_local_mask(self, is_lazy_mask: bool)` — [`L1075`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L1075)
  - `test_local_mask_narrow(self, is_lazy_mask: bool)` — [`L1174`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L1174)
  - `test_no_partial_mask_blocks(self)` — [`L780`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L780)
  - `test_process_invalid_mask(self)` — [`L1662`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L1662) — Masks with of an all-0 row causes undefined softmax, reject them.
  - `test_rectangular_tall_causal_mask(self, is_lazy_mask: bool)` — [`L940`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L940)
  - `test_rectangular_wide_causal_mask(self, is_lazy_mask: bool, return_dynamic_grid: bool)` — [`L831`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L831)
  - `test_two_qseq_shards_causal_local_stacked(self)` — [`L1240`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L1240)
  - `test_two_shards_local_wide_local_narrow_stacked(self, q_seq_shards, kv_seq_shards)` — [`L1376`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L1376)
- protocol/private: `_assert_mask_info_match`[`L687`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L687), `_check_presence`[`L690`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L690), `_process_mask`[`L744`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L744)
- uses (calls/refs, reference-scoped): [`_assert_array_equal`](splash_attention_test_utils.md#SplashAttentionTestCase._assert_array_equal), [`MaskInfo`](splash_attention_mask_info.md#MaskInfo), [`block_mask`](splash_attention_mask_info.md#MaskInfo.block_mask), [`partial_mask_blocks`](splash_attention_mask_info.md#MaskInfo.partial_mask_blocks), [`q_sequence`](splash_attention_mask_info.md#MaskInfo.q_sequence), [`active_rows`](splash_attention_mask_info.md#MaskInfo.active_rows), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`mask_next`](splash_attention_mask_info.md#MaskInfo.mask_next), [`active_cols`](splash_attention_mask_info.md#MaskInfo.active_cols), [`num_active_blocks`](splash_attention_mask_info.md#MaskInfo.num_active_blocks), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase), [`CausalMask`](splash_attention_mask.md#CausalMask), [`process_mask`](splash_attention_mask_info.md#process_mask), [`make_causal_mask`](splash_attention_mask.md#make_causal_mask), [`FullMask`](splash_attention_mask.md#FullMask), [`LocalMask`](splash_attention_mask.md#LocalMask), [`make_local_attention_mask`](splash_attention_mask.md#make_local_attention_mask), [`_check_mask`](splash_attention_mask_info.md#_check_mask), [`process_mask_dkv`](splash_attention_mask_info.md#process_mask_dkv), [`_make_causal_mask`](splash_attention_mask_test.md#_make_causal_mask), [`process_dynamic_mask`](splash_attention_mask_info.md#process_dynamic_mask), [`find_bounds`](splash_attention_mask_info.md#find_bounds)
- used by: [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase)

### `SplashAttentionMaskTest`  ·  implements/extends SplashAttentionTestCase
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py:59`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L59)
- signature: `class SplashAttentionMaskTest(test_utils.SplashAttentionTestCase):`
- members:
  - `setUp(self)` — [`L61`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L61)
  - `test_causal_mask(self, make_causal_mask)` — [`L67`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L67)
  - `test_chunked_causal_mask(self, make_chunked_mask)` — [`L435`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L435) — Tests the chunked causal mask logic for various shapes and chunk sizes.
  - `test_chunked_causal_mask_invalid_chunk_size(self)` — [`L567`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L567) — Tests that invalid chunk_size raises ValueError.
  - `test_chunked_causal_mask_minimal_equality_hash(self)` — [`L576`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L576) — Tests for __eq__ and __hash__ of ChunkedCausalMask.
  - `test_lazy_causal_mask_chunking(self, block_size: tuple[int, int], shape: tuple[int, int])` — [`L375`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L375)
  - `test_lazy_chunked_causal_mask_chunking(self, block_size: tuple[int, int], shape: tuple[int, int], chunk_size: int)` — [`L537`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L537) — Compares lazy chunked mask evaluation against the dense version block-by-block.
  - `test_lazy_full_mask(self, shape: tuple[int, int])` — [`L650`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L650)
  - `test_lazy_local_mask_chunking(self, block_size: tuple[int, int], shape: tuple[int, int], window_size: tuple[int | None, int | None], offset: int)` — [`L416`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L416)
  - `test_lazy_mask_and(self, shape: tuple[int, int])` — [`L640`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L640)
  - `test_lazy_mask_or(self, shape: tuple[int, int])` — [`L630`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L630)
  - `test_local_attention_mask(self, make_local_attention_mask)` — [`L149`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L149)
  - `test_local_attention_mask_tall_rectangle(self, make_local_attention_mask)` — [`L295`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L295)
  - `test_local_attention_mask_wide_rectangle(self, make_local_attention_mask)` — [`L226`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L226)
  - `test_using_logical_operators_raises_exception(self)` — [`L607`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L607)
- protocol/private: `_compare_masks`[`L656`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L656)
- uses (calls/refs, reference-scoped): [`_assert_array_equal`](splash_attention_test_utils.md#SplashAttentionTestCase._assert_array_equal), [`Mask`](splash_attention_mask.md#Mask), [`shape`](splash_attention_mask.md#Mask.shape), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase), [`ChunkedCausalMask`](splash_attention_mask.md#ChunkedCausalMask), [`CausalMask`](splash_attention_mask.md#CausalMask), [`make_causal_mask`](splash_attention_mask.md#make_causal_mask), [`FullMask`](splash_attention_mask.md#FullMask), [`LocalMask`](splash_attention_mask.md#LocalMask), [`make_local_attention_mask`](splash_attention_mask.md#make_local_attention_mask), [`make_random_mask`](splash_attention_mask.md#make_random_mask), [`_make_lazy_local_attention_mask`](splash_attention_mask_test.md#_make_lazy_local_attention_mask), [`_make_local_attention_mask`](splash_attention_mask_test.md#_make_local_attention_mask), [`_make_causal_mask`](splash_attention_mask_test.md#_make_causal_mask), [`_make_chunked_causal_mask`](splash_attention_mask_test.md#_make_chunked_causal_mask), [`_make_lazy_causal_mask`](splash_attention_mask_test.md#_make_lazy_causal_mask), [`_make_lazy_chunked_causal_mask`](splash_attention_mask_test.md#_make_lazy_chunked_causal_mask), [`make_chunk_attention_mask`](splash_attention_mask.md#make_chunk_attention_mask)
- used by: [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase)

## Functions
- `_make_causal_mask(*args, **kwargs)` — [`L37`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L37)
- `_make_chunked_causal_mask(shape, chunk_size)` — [`L55`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L55)
- `_make_lazy_causal_mask(*args, **kwargs)` — [`L32`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L32)
- `_make_lazy_chunked_causal_mask(shape, chunk_size)` — [`L50`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L50)
- `_make_lazy_local_attention_mask(*args, **kwargs)` — [`L41`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L41)
- `_make_local_attention_mask(*args, **kwargs)` — [`L46`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py#L46)

