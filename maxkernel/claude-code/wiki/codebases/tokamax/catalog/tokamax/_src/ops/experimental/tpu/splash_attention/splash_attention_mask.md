---
title: 'Module: tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_mask`/
symbols:
  Mask: Mask#
  Mask.shape: Mask#shape().
  NumpyMask: NumpyMask#
  _ComputableMask.shape: _ComputableMask#shape().
  _ComputableMask.q_sequence: _ComputableMask#q_sequence.
  _ComputableMask: _ComputableMask#
  _ComputableMask.__init__: _ComputableMask#__init__().
  ChunkedCausalMask: ChunkedCausalMask#
  LogicalOr.__init__: LogicalOr#__init__().
  LogicalAnd.__init__: LogicalAnd#__init__().
  _ComputableMask.__getitem__: _ComputableMask#__getitem__().
  LocalMask.__init__: LocalMask#__init__().
  LocalMask.__eq__: LocalMask#__eq__().
  LocalMask.__hash__: LocalMask#__hash__().
  CausalMask: CausalMask#
  FullMask.shape: FullMask#shape().
  make_causal_mask: make_causal_mask().
  LocalMask: LocalMask#
  FullMask: FullMask#
  CausalMask.__init__: CausalMask#__init__().
  CausalMask.__eq__: CausalMask#__eq__().
  CausalMask.__hash__: CausalMask#__hash__().
  ChunkedCausalMask.__init__: ChunkedCausalMask#__init__().
  ChunkedCausalMask.__eq__: ChunkedCausalMask#__eq__().
  ChunkedCausalMask.__hash__: ChunkedCausalMask#__hash__().
  make_local_attention_mask: make_local_attention_mask().
  make_random_mask: make_random_mask().
  NumpyMask.array: NumpyMask#array.
  LogicalOr.left: LogicalOr#left.
  LogicalAnd.left: LogicalAnd#left.
  CausalMask.offset: CausalMask#offset.
  ChunkedCausalMask.chunk_size: ChunkedCausalMask#chunk_size.
  LocalMask.local_mask_function: LocalMask#local_mask_function().
  LogicalOr.right: LogicalOr#right.
  LogicalAnd.right: LogicalAnd#right.
  Mask.__or__: Mask#__or__().
  Mask.__and__: Mask#__and__().
  LogicalOr.shape: LogicalOr#shape().
  LogicalOr.__getitem__: LogicalOr#__getitem__().
  LogicalOr.__hash__: LogicalOr#__hash__().
  LogicalAnd.shape: LogicalAnd#shape().
  LogicalAnd.__getitem__: LogicalAnd#__getitem__().
  LogicalAnd.__hash__: LogicalAnd#__hash__().
  LocalMask.window_size: LocalMask#window_size.
  LocalMask.offset: LocalMask#offset.
  FullMask.__getitem__: FullMask#__getitem__().
  LogicalOr: LogicalOr#
  LogicalAnd: LogicalAnd#
  _fill_slice: _fill_slice().
  CausalMask.causal_mask_function: CausalMask#causal_mask_function().
  ChunkedCausalMask.chunked_causal_mask_function: ChunkedCausalMask#chunked_causal_mask_function().
  NumpyMask.__post_init__: NumpyMask#__post_init__().
  NumpyMask.shape: NumpyMask#shape().
  NumpyMask.__getitem__: NumpyMask#__getitem__().
  NumpyMask.__eq__: NumpyMask#__eq__().
  NumpyMask.__hash__: NumpyMask#__hash__().
  FullMask.__post_init__: FullMask#__post_init__().
  FullMask.__eq__: FullMask#__eq__().
  FullMask.__hash__: FullMask#__hash__().
  make_chunk_attention_mask: make_chunk_attention_mask().
  _ComputableMask._shape: _ComputableMask#_shape.
  _ComputableMask.mask_function: _ComputableMask#mask_function.
  FullMask._shape: FullMask#_shape.
  Mask.__getitem__: Mask#__getitem__().
  Mask.__bool__: Mask#__bool__().
  _ComputableMask.__eq__: _ComputableMask#__eq__().
  _ComputableMask.__hash__: _ComputableMask#__hash__().
---
# Module: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py)

## Classes
### `CausalMask`  ·  implements/extends _ComputableMask
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py:247`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L247) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md)
- doc: Lazy causal mask, prevents the model from attending to future tokens.
- signature: `class CausalMask(_ComputableMask):`
- members:
  - `causal_mask_function(q_ids, kv_ids)` — [`L267`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L267)
  - `offset` — [`L257`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L257)
- protocol/private: `__eq__`[`L284`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L284), `__hash__`[`L294`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L294), `__init__`[`L259`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L259)
- uses (calls/refs, reference-scoped): [`shape`](splash_attention_mask.md#_ComputableMask.shape), [`q_sequence`](splash_attention_mask.md#_ComputableMask.q_sequence), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`__init__`](splash_attention_mask.md#_ComputableMask.__init__)
- used by: [`build_splash_kernel`](../../../attention/pallas_mosaic_tpu_common.md#build_splash_kernel), [`test_rectangular_wide_causal_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_rectangular_wide_causal_mask), [`test_huge_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_huge_mask), [`test_rectangular_tall_causal_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_rectangular_tall_causal_mask), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`get_mask`](splash_attention_kernel_test.md#CausalMask.get_mask), [`test_lazy_causal_mask_chunking`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_lazy_causal_mask_chunking), [`_make_lazy_causal_mask`](splash_attention_mask_test.md#_make_lazy_causal_mask)

### `ChunkedCausalMask`  ·  implements/extends _ComputableMask
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py:303`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L303) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md)
- doc: Lazy chunked causal mask.
- signature: `class ChunkedCausalMask(_ComputableMask):`
- members:
  - `chunked_causal_mask_function(q_ids, kv_ids)` — [`L328`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L328) — Computes the mask logic for the given slice indices.
  - `chunk_size` — [`L315`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L315)
- protocol/private: `__eq__`[`L344`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L344), `__hash__`[`L354`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L354), `__init__`[`L317`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L317)
- uses (calls/refs, reference-scoped): [`shape`](splash_attention_mask.md#_ComputableMask.shape), [`q_sequence`](splash_attention_mask.md#_ComputableMask.q_sequence), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`__init__`](splash_attention_mask.md#_ComputableMask.__init__)
- used by: [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`test_lazy_chunked_causal_mask_chunking`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_lazy_chunked_causal_mask_chunking), [`test_chunked_causal_mask_invalid_chunk_size`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_chunked_causal_mask_invalid_chunk_size), [`_make_lazy_chunked_causal_mask`](splash_attention_mask_test.md#_make_lazy_chunked_causal_mask), [`test_chunked_causal_mask_minimal_equality_hash`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_chunked_causal_mask_minimal_equality_hash)

### `FullMask`  ·  implements/extends Mask
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py:481`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L481) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md)
- doc: Lazy full mask, allows all tokens to attend to all other tokens.
- signature: `class FullMask(Mask):`
- members:
  - `shape(self)` — [`L493`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L493)
- protocol/private: `__eq__`[`L506`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L506), `__getitem__`[`L496`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L496), `__hash__`[`L512`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L512), `__post_init__`[`L488`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L488), `_shape`[`L486`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L486)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`_fill_slice`](splash_attention_mask.md#_fill_slice)
- used by: [`make_ring_attention`](ring_attention_kernel.md#make_ring_attention), [`test_ring_attention`](ring_attention_kernel_test.md#RingAttentionTest.test_ring_attention), [`Mask`](splash_attention_mask.md#Mask), [`shape`](splash_attention_mask.md#Mask.shape), [`build_splash_kernel`](../../../attention/pallas_mosaic_tpu_common.md#build_splash_kernel), [`test_full_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_full_mask), [`get_mask`](splash_attention_kernel_test.md#FullMask.get_mask), [`test_lazy_full_mask`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_lazy_full_mask)

### `LocalMask`  ·  implements/extends _ComputableMask
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py:363`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L363) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md)
- doc: Lazy local mask, prevents model from attending to tokens outside window.
- signature: `class LocalMask(_ComputableMask):`
- members:
  - `local_mask_function(q_ids, kv_ids)` — [`L388`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L388) — Computes the local attention mask for the given slice indices.
  - `offset` — [`L376`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L376)
  - `window_size` — [`L375`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L375)
- protocol/private: `__eq__`[`L420`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L420), `__hash__`[`L431`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L431), `__init__`[`L378`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L378)
- uses (calls/refs, reference-scoped): [`shape`](splash_attention_mask.md#_ComputableMask.shape), [`q_sequence`](splash_attention_mask.md#_ComputableMask.q_sequence), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`__init__`](splash_attention_mask.md#_ComputableMask.__init__)
- used by: [`get_mask`](splash_attention_kernel_test.md#LocalAttentionMask.get_mask), [`test_huge_mask2`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_huge_mask2), [`test_local_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_local_mask), [`test_local_mask_narrow`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_local_mask_narrow), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`test_lazy_local_mask_chunking`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_lazy_local_mask_chunking), [`_make_lazy_local_attention_mask`](splash_attention_mask_test.md#_make_lazy_local_attention_mask)

### `LogicalAnd`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py:156`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L156) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md)
- members:
  - `shape(self)` — [`L167`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L167)
  - `left` — [`L157`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L157) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md)
  - `right` — [`L158`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L158) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md)
- protocol/private: `__getitem__`[`L170`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L170), `__hash__`[`L173`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L173), `__init__`[`L160`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L160)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`shape`](splash_attention_mask.md#Mask.shape)
- used by: [`shape`](splash_attention_mask.md#Mask.shape), [`compose_mask_strategy`](splash_attention_kernel_test.md#compose_mask_strategy), [`__and__`](splash_attention_mask.md#Mask.__and__)

### `LogicalOr`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py:134`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L134)
- members:
  - `shape(self)` — [`L145`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L145)
  - `left` — [`L135`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L135)
  - `right` — [`L136`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L136)
- protocol/private: `__getitem__`[`L148`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L148), `__hash__`[`L151`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L151), `__init__`[`L138`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L138)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`shape`](splash_attention_mask.md#Mask.shape)
- used by: [`shape`](splash_attention_mask.md#Mask.shape), [`compose_mask_strategy`](splash_attention_kernel_test.md#compose_mask_strategy), [`__or__`](splash_attention_mask.md#Mask.__or__)

### `Mask`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py:27`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L27) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md)
- doc: A base class for splash attention masks.
- signature: `class Mask:`
- members:
  - `shape(self)` — [`L31`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L31) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md)
- protocol/private: `__and__`[`L50`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L50), `__bool__`[`L37`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L37), `__getitem__`[`L34`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L34), `__or__`[`L43`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L43)
- uses (calls/refs, reference-scoped): [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`shape`](splash_attention_mask.md#_ComputableMask.shape), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`shape`](splash_attention_mask.md#FullMask.shape), [`FullMask`](splash_attention_mask.md#FullMask), [`shape`](splash_attention_mask.md#LogicalAnd.shape), [`shape`](splash_attention_mask.md#LogicalOr.shape), [`LogicalAnd`](splash_attention_mask.md#LogicalAnd), [`LogicalOr`](splash_attention_mask.md#LogicalOr), [`shape`](splash_attention_mask.md#NumpyMask.shape)
- used by: [`make_ring_attention`](ring_attention_kernel.md#make_ring_attention), [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`_process_mask`](splash_attention_mask_info.md#_process_mask), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`get_mask`](splash_attention_kernel_test.md#LocalAttentionMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#RandomMask.get_mask), [`_compare_masks`](splash_attention_mask_test.md#SplashAttentionMaskTest._compare_masks), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`check_mask_no_empty_rows`](splash_attention_kernel_test.md#check_mask_no_empty_rows), [`get_mask`](splash_attention_kernel_test.md#ComposeMask.get_mask), [`__init__`](splash_attention_mask.md#LogicalAnd.__init__), [`__init__`](splash_attention_mask.md#LogicalOr.__init__), [`get_mask`](splash_attention_kernel_test.md#CausalMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#FullMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#SplitMask.get_mask), [`FullMask`](splash_attention_mask.md#FullMask), [`left`](splash_attention_mask.md#LogicalAnd.left), [`left`](splash_attention_mask.md#LogicalOr.left), [`_check_mask`](splash_attention_mask_info.md#_check_mask), [`right`](splash_attention_mask.md#LogicalAnd.right), [`right`](splash_attention_mask.md#LogicalOr.right), [`shape`](splash_attention_mask.md#LogicalAnd.shape), [`shape`](splash_attention_mask.md#LogicalOr.shape), [`LogicalAnd`](splash_attention_mask.md#LogicalAnd), [`LogicalOr`](splash_attention_mask.md#LogicalOr), [`op`](splash_attention_kernel_test.md#ComposeMask.op)

### `NumpyMask`  ·  implements/extends Mask
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py:442`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L442) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_mask.md)
- doc: A mask backed by a dense numpy array.
- signature: `class NumpyMask(Mask):`
- members:
  - `shape(self)` — [`L455`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L455)
  - `array` — [`L445`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L445)
- protocol/private: `__eq__`[`L461`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L461), `__getitem__`[`L458`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L458), `__hash__`[`L467`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L467), `__post_init__`[`L447`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L447)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask)
- used by: [`make_ring_attention`](ring_attention_kernel.md#make_ring_attention), [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`Mask`](splash_attention_mask.md#Mask), [`shape`](splash_attention_mask.md#Mask.shape), [`test_causal_two_q_shards_two_kv_shards`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_causal_two_q_shards_two_kv_shards), [`test_no_partial_mask_blocks`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_no_partial_mask_blocks), [`get_mask`](splash_attention_kernel_test.md#LocalAttentionMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#RandomMask.get_mask), [`test_rectangular_wide_causal_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_rectangular_wide_causal_mask), [`test_local_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_local_mask), [`test_local_mask_narrow`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_local_mask_narrow), [`test_rectangular_tall_causal_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_rectangular_tall_causal_mask), [`test_two_qseq_shards_causal_local_stacked`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_two_qseq_shards_causal_local_stacked), [`test_full_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_full_mask), [`test_two_shards_local_wide_local_narrow_stacked`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_two_shards_local_wide_local_narrow_stacked), [`get_mask`](splash_attention_kernel_test.md#SplitMask.get_mask), [`test_lazy_mask_and`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_lazy_mask_and), [`test_lazy_mask_or`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_lazy_mask_or), [`test_process_invalid_mask`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest.test_process_invalid_mask), [`test_using_logical_operators_raises_exception`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_using_logical_operators_raises_exception)

### `_ComputableMask`  ·  implements/extends Mask
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py:177`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L177)
- doc: Superclass for all masks that can be computed inside the kernel using a callable object.
- signature: `class _ComputableMask(Mask):`
- members:
  - `shape(self)` — [`L221`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L221)
  - `mask_function` — [`L200`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L200)
  - `q_sequence` — [`L199`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L199)
- protocol/private: `__eq__`[`L240`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L240), `__getitem__`[`L224`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L224), `__hash__`[`L243`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L243), `__init__`[`L202`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L202), `_shape`[`L198`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L198)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`ChunkedCausalMask`](splash_attention_mask.md#ChunkedCausalMask), [`CausalMask`](splash_attention_mask.md#CausalMask), [`LocalMask`](splash_attention_mask.md#LocalMask), [`_fill_slice`](splash_attention_mask.md#_fill_slice)
- used by: [`Mask`](splash_attention_mask.md#Mask), [`shape`](splash_attention_mask.md#Mask.shape), [`ChunkedCausalMask`](splash_attention_mask.md#ChunkedCausalMask), [`__eq__`](splash_attention_mask.md#LocalMask.__eq__), [`__hash__`](splash_attention_mask.md#LocalMask.__hash__), [`__init__`](splash_attention_mask.md#LocalMask.__init__), [`CausalMask`](splash_attention_mask.md#CausalMask), [`LocalMask`](splash_attention_mask.md#LocalMask), [`__eq__`](splash_attention_mask.md#CausalMask.__eq__), [`__eq__`](splash_attention_mask.md#ChunkedCausalMask.__eq__), [`__hash__`](splash_attention_mask.md#CausalMask.__hash__), [`__hash__`](splash_attention_mask.md#ChunkedCausalMask.__hash__), [`__init__`](splash_attention_mask.md#CausalMask.__init__), [`__init__`](splash_attention_mask.md#ChunkedCausalMask.__init__)

## Functions
- `_fill_slice(inp_slice: slice, size: int)` — [`L471`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L471)
- `make_causal_mask(shape: tuple[int, int], offset: int = 0)` — [`L58`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L58) — Makes a causal attention mask.
- `make_chunk_attention_mask(shape: tuple[int, int], chunk_size: int)` — [`L96`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L96) — Makes a chunked causal attention mask.
- `make_local_attention_mask(shape: tuple[int, int], window_size: tuple[int | None, int | None], *, offset: int = 0)` — [`L77`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L77) — Makes a local attention mask.
- `make_random_mask(shape: tuple[int, int], sparsity: float, seed: int)` — [`L125`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py#L125) — Makes a random attention mask.

