---
title: 'Module: ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.blocksparse_attention._masks`/
symbols:
  Mask: Mask#
  Mask.shape: Mask#shape().
  MultiHeadMask.masks: MultiHeadMask#masks.
  _ComputableMask.shape: _ComputableMask#shape().
  _ComputableMask.q_sequence: _ComputableMask#q_sequence.
  _ComputableMask: _ComputableMask#
  _ComputableMask.__init__: _ComputableMask#__init__().
  MultiHeadMask: MultiHeadMask#
  LogicalOr.__init__: LogicalOr#__init__().
  LogicalAnd.__init__: LogicalAnd#__init__().
  MultiHeadMask.__post_init__: MultiHeadMask#__post_init__().
  _ComputableMask.__getitem__: _ComputableMask#__getitem__().
  LocalMask.__init__: LocalMask#__init__().
  LocalMask.__eq__: LocalMask#__eq__().
  LocalMask.__hash__: LocalMask#__hash__().
  FullMask.shape: FullMask#shape().
  MultiHeadMask.shape: MultiHeadMask#shape().
  Mask.__or__: Mask#__or__().
  Mask.__and__: Mask#__and__().
  CausalMask.__init__: CausalMask#__init__().
  CausalMask.__eq__: CausalMask#__eq__().
  CausalMask.__hash__: CausalMask#__hash__().
  ChunkedCausalMask.__init__: ChunkedCausalMask#__init__().
  ChunkedCausalMask.__eq__: ChunkedCausalMask#__eq__().
  ChunkedCausalMask.__hash__: ChunkedCausalMask#__hash__().
  NumpyMask.array: NumpyMask#array.
  LogicalOr.left: LogicalOr#left.
  LogicalAnd.left: LogicalAnd#left.
  CausalMask: CausalMask#
  CausalMask.offset: CausalMask#offset.
  ChunkedCausalMask.chunk_size: ChunkedCausalMask#chunk_size.
  LocalMask.local_mask_function: LocalMask#local_mask_function().
  LogicalOr.right: LogicalOr#right.
  LogicalAnd.right: LogicalAnd#right.
  ChunkedCausalMask: ChunkedCausalMask#
  LocalMask: LocalMask#
  NumpyMask: NumpyMask#
  FullMask: FullMask#
  LogicalOr.shape: LogicalOr#shape().
  LogicalOr.__getitem__: LogicalOr#__getitem__().
  LogicalOr.__hash__: LogicalOr#__hash__().
  LogicalAnd.shape: LogicalAnd#shape().
  LogicalAnd.__getitem__: LogicalAnd#__getitem__().
  LogicalAnd.__hash__: LogicalAnd#__hash__().
  LocalMask.window_size: LocalMask#window_size.
  LocalMask.offset: LocalMask#offset.
  FullMask.__getitem__: FullMask#__getitem__().
  _fill_slice: _fill_slice().
  CausalMask.causal_mask_function: CausalMask#causal_mask_function().
  ChunkedCausalMask.chunked_causal_mask_function: ChunkedCausalMask#chunked_causal_mask_function().
  LogicalOr: LogicalOr#
  LogicalAnd: LogicalAnd#
  MultiHeadMask.__getitem__: MultiHeadMask#__getitem__().
  MultiHeadMask.__eq__: MultiHeadMask#__eq__().
  MultiHeadMask.__hash__: MultiHeadMask#__hash__().
  NumpyMask.__post_init__: NumpyMask#__post_init__().
  NumpyMask.shape: NumpyMask#shape().
  NumpyMask.__getitem__: NumpyMask#__getitem__().
  NumpyMask.__eq__: NumpyMask#__eq__().
  NumpyMask.__hash__: NumpyMask#__hash__().
  FullMask.__post_init__: FullMask#__post_init__().
  FullMask.__eq__: FullMask#__eq__().
  FullMask.__hash__: FullMask#__hash__().
  _ComputableMask._shape: _ComputableMask#_shape.
  _ComputableMask.mask_function: _ComputableMask#mask_function.
  make_causal_mask: make_causal_mask().
  make_local_attention_mask: make_local_attention_mask().
  make_chunk_attention_mask: make_chunk_attention_mask().
  make_random_mask: make_random_mask().
  FullMask._shape: FullMask#_shape.
  Mask.__getitem__: Mask#__getitem__().
  Mask.__bool__: Mask#__bool__().
  _ComputableMask.__eq__: _ComputableMask#__eq__().
  _ComputableMask.__hash__: _ComputableMask#__hash__().
---
# Module: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py)

## Classes
### `CausalMask`  ·  implements/extends _ComputableMask
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py:384`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L384) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
- doc: Lazy lower-triangular causal mask for autoregressive attention.
- signature: `class CausalMask(_ComputableMask):`
- members:
  - `causal_mask_function(q_ids, kv_ids)` — [`L410`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L410)
  - `offset` — [`L400`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L400)
- protocol/private: `__eq__`[`L424`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L424), `__hash__`[`L434`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L434), `__init__`[`L402`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L402)
- uses (calls/refs, reference-scoped): [`shape`](_masks.md#_ComputableMask.shape), [`q_sequence`](_masks.md#_ComputableMask.q_sequence), [`_ComputableMask`](_masks.md#_ComputableMask), [`__init__`](_masks.md#_ComputableMask.__init__)
- used by: [`blocksparse_attention`](_kernel.md#blocksparse_attention), [`_ComputableMask`](_masks.md#_ComputableMask)

### `ChunkedCausalMask`  ·  implements/extends _ComputableMask
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py:445`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L445) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
- doc: Lazy chunked causal mask for block-diagonal attention (Llama4-style).
- signature: `class ChunkedCausalMask(_ComputableMask):`
- members:
  - `chunked_causal_mask_function(q_ids, kv_ids)` — [`L473`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L473) — Computes the mask logic for the given slice indices.
  - `chunk_size` — [`L461`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L461)
- protocol/private: `__eq__`[`L488`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L488), `__hash__`[`L498`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L498), `__init__`[`L463`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L463)
- uses (calls/refs, reference-scoped): [`shape`](_masks.md#_ComputableMask.shape), [`q_sequence`](_masks.md#_ComputableMask.q_sequence), [`_ComputableMask`](_masks.md#_ComputableMask), [`__init__`](_masks.md#_ComputableMask.__init__)
- used by: [`blocksparse_attention`](_kernel.md#blocksparse_attention), [`_ComputableMask`](_masks.md#_ComputableMask)

### `FullMask`  ·  implements/extends Mask
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py:654`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L654) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
- doc: Lazy dense (all-ones) mask — every token can attend every other token.
- signature: `class FullMask(Mask):`
- members:
  - `shape(self)` — [`L672`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L672) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
- protocol/private: `__eq__`[`L685`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L685), `__getitem__`[`L675`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L675), `__hash__`[`L691`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L691), `__post_init__`[`L667`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L667), `_shape`[`L665`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L665)
- uses (calls/refs, reference-scoped): [`Mask`](_masks.md#Mask), [`_fill_slice`](_masks.md#_fill_slice)
- used by: [`blocksparse_attention`](_kernel.md#blocksparse_attention), [`Mask`](_masks.md#Mask), [`shape`](_masks.md#Mask.shape)

### `LocalMask`  ·  implements/extends _ComputableMask
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py:509`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L509) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
- doc: Lazy sliding-window local attention mask.
- signature: `class LocalMask(_ComputableMask):`
- members:
  - `local_mask_function(q_ids, kv_ids)` — [`L540`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L540) — Computes the local attention mask for the given slice indices.
  - `offset` — [`L528`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L528)
  - `window_size` — [`L527`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L527)
- protocol/private: `__eq__`[`L571`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L571), `__hash__`[`L582`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L582), `__init__`[`L530`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L530)
- uses (calls/refs, reference-scoped): [`shape`](_masks.md#_ComputableMask.shape), [`q_sequence`](_masks.md#_ComputableMask.q_sequence), [`_ComputableMask`](_masks.md#_ComputableMask), [`__init__`](_masks.md#_ComputableMask.__init__)
- used by: [`blocksparse_attention`](_kernel.md#blocksparse_attention), [`_ComputableMask`](_masks.md#_ComputableMask)

### `LogicalAnd`
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py:227`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L227) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
- members:
  - `__getitem__(self, idx)` — [`L252`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L252) — Return element-wise AND of both masks at the given indices.
  - `shape(self)` — [`L248`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L248) — Return the shape of the combined mask. — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
  - `left` — [`L238`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L238) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
  - `right` — [`L239`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L239) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
- protocol/private: `__hash__`[`L256`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L256), `__init__`[`L241`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L241)
- uses (calls/refs, reference-scoped): [`Mask`](_masks.md#Mask), [`shape`](_masks.md#Mask.shape)
- used by: [`shape`](_masks.md#Mask.shape), [`__and__`](_masks.md#Mask.__and__)

### `LogicalOr`
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py:193`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L193) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
- members:
  - `__getitem__(self, idx)` — [`L218`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L218) — Return element-wise OR of both masks at the given indices.
  - `shape(self)` — [`L214`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L214) — Return the shape of the combined mask. — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
  - `left` — [`L204`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L204)
  - `right` — [`L205`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L205)
- protocol/private: `__hash__`[`L222`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L222), `__init__`[`L207`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L207)
- uses (calls/refs, reference-scoped): [`Mask`](_masks.md#Mask), [`shape`](_masks.md#Mask.shape)
- used by: [`shape`](_masks.md#Mask.shape), [`__or__`](_masks.md#Mask.__or__)

### `Mask`
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py:57`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L57) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
- doc: Base class for block-sparse attention masks.
- signature: `class Mask:`
- members:
  - `__and__(self, other: Mask)` — [`L89`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L89) — Combine masks with logical AND (both masks must allow attention). — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
  - `__getitem__(self, idx)` — [`L74`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L74) — Return mask values for the given slice indices.
  - `__or__(self, other: Mask)` — [`L83`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L83) — Combine masks with logical OR (either mask allows attention). — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md)
  - `shape(self)` — [`L70`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L70) — Return the shape of the mask (q_seq_len, kv_seq_len).
- protocol/private: `__bool__`[`L78`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L78)
- uses (calls/refs, reference-scoped): [`shape`](_masks.md#_ComputableMask.shape), [`_ComputableMask`](_masks.md#_ComputableMask), [`MultiHeadMask`](_masks.md#MultiHeadMask), [`shape`](_masks.md#FullMask.shape), [`shape`](_masks.md#MultiHeadMask.shape), [`FullMask`](_masks.md#FullMask), [`NumpyMask`](_masks.md#NumpyMask), [`shape`](_masks.md#LogicalAnd.shape), [`shape`](_masks.md#LogicalOr.shape), [`LogicalAnd`](_masks.md#LogicalAnd), [`LogicalOr`](_masks.md#LogicalOr), [`shape`](_masks.md#NumpyMask.shape)
- used by: [`blocksparse_attention`](_kernel.md#blocksparse_attention), [`_process_mask`](_info.md#_process_mask), [`make_ring_attention`](../ring_attention/_pallas_impl_bwd.md#make_ring_attention), [`ring_attention`](../ring_attention/_pallas_impl_fwd.md#ring_attention), [`blocksparse_attention`](../../../../modules/operations/blocksparse_attention.md#blocksparse_attention), [`ring_attention`](../../../_xla/ring_attention/_interface.md#ring_attention), [`ring_attention`](../../../../modules/operations/ring_attention.md#ring_attention), [`ring_attention`](../ring_attention/_interface.md#ring_attention), [`run`](../../../../modules/operations/blocksparse_attention.md#BlockSparseAttention.run), [`run`](../../../../modules/operations/ring_attention.md#RingAttention.run), [`masks`](_masks.md#MultiHeadMask.masks), [`_ComputableMask`](_masks.md#_ComputableMask), [`MultiHeadMask`](_masks.md#MultiHeadMask), [`__init__`](_masks.md#LogicalAnd.__init__), [`__init__`](_masks.md#LogicalOr.__init__), [`__post_init__`](_masks.md#MultiHeadMask.__post_init__), [`create_shard_map_wrapper`](../../../../modules/operations/blocksparse_attention.md#BlockSparseAttention.create_shard_map_wrapper), [`shape`](_masks.md#MultiHeadMask.shape), [`make_attention_reference`](_kernel.md#make_attention_reference), [`create_shard_map_wrapper`](../../../../modules/operations/ring_attention.md#RingAttention.create_shard_map_wrapper), [`left`](_masks.md#LogicalAnd.left), [`left`](_masks.md#LogicalOr.left), [`FullMask`](_masks.md#FullMask), [`NumpyMask`](_masks.md#NumpyMask), [`right`](_masks.md#LogicalAnd.right), [`right`](_masks.md#LogicalOr.right), [`_check_mask`](_info.md#_check_mask), [`shape`](_masks.md#LogicalAnd.shape), [`shape`](_masks.md#LogicalOr.shape), [`LogicalAnd`](_masks.md#LogicalAnd), [`LogicalOr`](_masks.md#LogicalOr)

### `MultiHeadMask`  ·  implements/extends Mask
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py:261`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L261)
- doc: Per-head mask wrapper that combines multiple single-head masks.
- signature: `class MultiHeadMask(Mask):`
- members:
  - `shape(self)` — [`L293`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L293)
  - `masks` — [`L275`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L275)
- protocol/private: `__eq__`[`L308`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L308), `__getitem__`[`L296`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L296), `__hash__`[`L314`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L314), `__post_init__`[`L277`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L277)
- uses (calls/refs, reference-scoped): [`Mask`](_masks.md#Mask), [`shape`](_masks.md#Mask.shape)
- used by: [`blocksparse_attention`](_kernel.md#blocksparse_attention), [`Mask`](_masks.md#Mask), [`_process_mask`](_info.md#_process_mask), [`_make_splash_attention`](_kernel.md#_make_splash_attention), [`make_ring_attention`](../ring_attention/_pallas_impl_bwd.md#make_ring_attention), [`shape`](_masks.md#Mask.shape), [`_get_mask_info_for_shard`](_info.md#_get_mask_info_for_shard)

### `NumpyMask`  ·  implements/extends Mask
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py:595`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L595)
- doc: Dense attention mask backed by a numpy boolean array.
- signature: `class NumpyMask(Mask):`
- members:
  - `shape(self)` — [`L619`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L619)
  - `array` — [`L609`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L609)
- protocol/private: `__eq__`[`L625`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L625), `__getitem__`[`L622`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L622), `__hash__`[`L631`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L631), `__post_init__`[`L611`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L611)
- uses (calls/refs, reference-scoped): [`Mask`](_masks.md#Mask)
- used by: [`Mask`](_masks.md#Mask), [`_make_splash_attention`](_kernel.md#_make_splash_attention), [`make_ring_attention`](../ring_attention/_pallas_impl_bwd.md#make_ring_attention), [`shape`](_masks.md#Mask.shape)

### `_ComputableMask`  ·  implements/extends Mask
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py:318`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L318)
- doc: Base class for lazily-evaluated masks computed by a callable inside the kernel.
- signature: `class _ComputableMask(Mask):`
- members:
  - `shape(self)` — [`L358`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L358)
  - `mask_function` — [`L338`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L338)
  - `q_sequence` — [`L337`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L337)
- protocol/private: `__eq__`[`L377`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L377), `__getitem__`[`L361`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L361), `__hash__`[`L380`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L380), `__init__`[`L340`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L340), `_shape`[`L336`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L336)
- uses (calls/refs, reference-scoped): [`Mask`](_masks.md#Mask), [`CausalMask`](_masks.md#CausalMask), [`ChunkedCausalMask`](_masks.md#ChunkedCausalMask), [`LocalMask`](_masks.md#LocalMask), [`_fill_slice`](_masks.md#_fill_slice)
- used by: [`Mask`](_masks.md#Mask), [`shape`](_masks.md#Mask.shape), [`__eq__`](_masks.md#LocalMask.__eq__), [`__hash__`](_masks.md#LocalMask.__hash__), [`__init__`](_masks.md#LocalMask.__init__), [`__eq__`](_masks.md#CausalMask.__eq__), [`__eq__`](_masks.md#ChunkedCausalMask.__eq__), [`__hash__`](_masks.md#CausalMask.__hash__), [`__hash__`](_masks.md#ChunkedCausalMask.__hash__), [`__init__`](_masks.md#CausalMask.__init__), [`__init__`](_masks.md#ChunkedCausalMask.__init__), [`CausalMask`](_masks.md#CausalMask), [`ChunkedCausalMask`](_masks.md#ChunkedCausalMask), [`LocalMask`](_masks.md#LocalMask)

## Functions
- `_fill_slice(inp_slice: slice, size: int)` — [`L635`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L635) — Normalise a slice by filling in ``None`` start/stop with 0 / ``size``.
- `make_causal_mask(shape: tuple[int, int], offset: int = 0)` — [`L96`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L96) — Makes a causal attention mask.
- `make_chunk_attention_mask(shape: tuple[int, int], chunk_size: int)` — [`L148`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L148) — Makes a chunked causal attention mask.
- `make_local_attention_mask(shape: tuple[int, int], window_size: tuple[int | None, int | None], *, offset: int = 0)` — [`L115`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L115) — Create a local (sliding window) attention mask.
- `make_random_mask(shape: tuple[int, int], sparsity: float, seed: int)` — [`L174`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks.py#L174) — Create a random attention mask with specified sparsity.

