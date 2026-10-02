---
title: 'Module: jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask`/
symbols:
  Mask.shape: Mask#shape().
  Mask: Mask#
  MultiHeadMask.masks: MultiHeadMask#masks.
  _ComputableMask.shape: _ComputableMask#shape().
  _ComputableMask.q_sequence: _ComputableMask#q_sequence.
  _ComputableMask: _ComputableMask#
  _ComputableMask.__init__: _ComputableMask#__init__().
  LogicalOr.__init__: LogicalOr#__init__().
  LogicalAnd.__init__: LogicalAnd#__init__().
  MultiHeadMask.__post_init__: MultiHeadMask#__post_init__().
  _ComputableMask.__getitem__: _ComputableMask#__getitem__().
  LocalMask.__init__: LocalMask#__init__().
  LocalMask.__eq__: LocalMask#__eq__().
  LocalMask.__hash__: LocalMask#__hash__().
  MultiHeadMask: MultiHeadMask#
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
  CausalMask.offset: CausalMask#offset.
  ChunkedCausalMask.chunk_size: ChunkedCausalMask#chunk_size.
  LocalMask.local_mask_function: LocalMask#local_mask_function().
  LogicalOr.right: LogicalOr#right.
  LogicalAnd.right: LogicalAnd#right.
  LogicalOr.shape: LogicalOr#shape().
  LogicalOr.__getitem__: LogicalOr#__getitem__().
  LogicalOr.__hash__: LogicalOr#__hash__().
  LogicalAnd.shape: LogicalAnd#shape().
  LogicalAnd.__getitem__: LogicalAnd#__getitem__().
  LogicalAnd.__hash__: LogicalAnd#__hash__().
  LocalMask.window_size: LocalMask#window_size.
  LocalMask.offset: LocalMask#offset.
  FullMask.__getitem__: FullMask#__getitem__().
  NumpyMask: NumpyMask#
  _fill_slice: _fill_slice().
  CausalMask: CausalMask#
  CausalMask.causal_mask_function: CausalMask#causal_mask_function().
  ChunkedCausalMask.chunked_causal_mask_function: ChunkedCausalMask#chunked_causal_mask_function().
  LocalMask: LocalMask#
  FullMask: FullMask#
  LogicalOr: LogicalOr#
  LogicalAnd: LogicalAnd#
  MultiHeadMask.__getitem__: MultiHeadMask#__getitem__().
  MultiHeadMask.__eq__: MultiHeadMask#__eq__().
  MultiHeadMask.__hash__: MultiHeadMask#__hash__().
  ChunkedCausalMask: ChunkedCausalMask#
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
  make_random_mask: make_random_mask().
  FullMask._shape: FullMask#_shape.
  Mask.__getitem__: Mask#__getitem__().
  Mask.__bool__: Mask#__bool__().
  make_chunk_attention_mask: make_chunk_attention_mask().
  _ComputableMask.__eq__: _ComputableMask#__eq__().
  _ComputableMask.__hash__: _ComputableMask#__hash__().
---
# Module: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py)

## Classes
### `CausalMask`  ·  implements/extends _ComputableMask
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py:294`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L294)
- doc: Lazy causal mask, prevents the model from attending to future tokens.
- signature: `class CausalMask(_ComputableMask):`
- members:
  - `causal_mask_function(q_ids, kv_ids)` — [`L314`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L314)
  - `offset` — [`L304`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L304)
- protocol/private: `__eq__`[`L331`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L331), `__hash__`[`L341`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L341), `__init__`[`L306`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L306)
- uses (calls/refs, reference-scoped): [`shape`](splash_attention_mask.md#_ComputableMask.shape), [`q_sequence`](splash_attention_mask.md#_ComputableMask.q_sequence), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`__init__`](splash_attention_mask.md#_ComputableMask.__init__)
- used by: [`_ComputableMask`](splash_attention_mask.md#_ComputableMask)

### `ChunkedCausalMask`  ·  implements/extends _ComputableMask
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py:350`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L350)
- doc: Lazy chunked causal mask.
- signature: `class ChunkedCausalMask(_ComputableMask):`
- members:
  - `chunked_causal_mask_function(q_ids, kv_ids)` — [`L375`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L375) — Computes the mask logic for the given slice indices.
  - `chunk_size` — [`L362`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L362)
- protocol/private: `__eq__`[`L391`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L391), `__hash__`[`L401`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L401), `__init__`[`L364`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L364)
- uses (calls/refs, reference-scoped): [`shape`](splash_attention_mask.md#_ComputableMask.shape), [`q_sequence`](splash_attention_mask.md#_ComputableMask.q_sequence), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`__init__`](splash_attention_mask.md#_ComputableMask.__init__)
- used by: [`_ComputableMask`](splash_attention_mask.md#_ComputableMask)

### `FullMask`  ·  implements/extends Mask
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py:528`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L528)
- doc: Lazy full mask, allows all tokens to attend to all other tokens.
- signature: `class FullMask(Mask):`
- members:
  - `shape(self)` — [`L540`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L540)
- protocol/private: `__eq__`[`L553`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L553), `__getitem__`[`L543`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L543), `__hash__`[`L559`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L559), `__post_init__`[`L535`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L535), `_shape`[`L533`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L533)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`_fill_slice`](splash_attention_mask.md#_fill_slice)
- used by: [`shape`](splash_attention_mask.md#Mask.shape), [`Mask`](splash_attention_mask.md#Mask)

### `LocalMask`  ·  implements/extends _ComputableMask
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py:410`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L410)
- doc: Lazy local mask, prevents model from attending to tokens outside window.
- signature: `class LocalMask(_ComputableMask):`
- members:
  - `local_mask_function(q_ids, kv_ids)` — [`L435`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L435) — Computes the local attention mask for the given slice indices.
  - `offset` — [`L423`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L423)
  - `window_size` — [`L422`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L422)
- protocol/private: `__eq__`[`L467`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L467), `__hash__`[`L478`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L478), `__init__`[`L425`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L425)
- uses (calls/refs, reference-scoped): [`shape`](splash_attention_mask.md#_ComputableMask.shape), [`q_sequence`](splash_attention_mask.md#_ComputableMask.q_sequence), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`__init__`](splash_attention_mask.md#_ComputableMask.__init__)
- used by: [`_ComputableMask`](splash_attention_mask.md#_ComputableMask)

### `LogicalAnd`
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py:154`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L154)
- members:
  - `shape(self)` — [`L165`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L165)
  - `left` — [`L155`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L155)
  - `right` — [`L156`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L156)
- protocol/private: `__getitem__`[`L168`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L168), `__hash__`[`L171`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L171), `__init__`[`L158`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L158)
- uses (calls/refs, reference-scoped): [`shape`](splash_attention_mask.md#Mask.shape), [`Mask`](splash_attention_mask.md#Mask)
- used by: [`shape`](splash_attention_mask.md#Mask.shape), [`__and__`](splash_attention_mask.md#Mask.__and__)

### `LogicalOr`
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py:132`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L132)
- members:
  - `shape(self)` — [`L143`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L143)
  - `left` — [`L133`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L133)
  - `right` — [`L134`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L134)
- protocol/private: `__getitem__`[`L146`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L146), `__hash__`[`L149`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L149), `__init__`[`L136`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L136)
- uses (calls/refs, reference-scoped): [`shape`](splash_attention_mask.md#Mask.shape), [`Mask`](splash_attention_mask.md#Mask)
- used by: [`shape`](splash_attention_mask.md#Mask.shape), [`__or__`](splash_attention_mask.md#Mask.__or__)

### `Mask`
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py:25`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L25)
- doc: A base class for splash attention masks.
- signature: `class Mask:`
- members:
  - `shape(self)` — [`L29`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L29)
- protocol/private: `__and__`[`L48`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L48), `__bool__`[`L35`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L35), `__getitem__`[`L32`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L32), `__or__`[`L41`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L41)
- uses (calls/refs, reference-scoped): [`shape`](splash_attention_mask.md#_ComputableMask.shape), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`MultiHeadMask`](splash_attention_mask.md#MultiHeadMask), [`shape`](splash_attention_mask.md#FullMask.shape), [`shape`](splash_attention_mask.md#MultiHeadMask.shape), [`shape`](splash_attention_mask.md#LogicalAnd.shape), [`shape`](splash_attention_mask.md#LogicalOr.shape), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`FullMask`](splash_attention_mask.md#FullMask), [`LogicalAnd`](splash_attention_mask.md#LogicalAnd), [`LogicalOr`](splash_attention_mask.md#LogicalOr), [`shape`](splash_attention_mask.md#NumpyMask.shape)
- used by: [`_process_mask`](splash_attention_mask_info.md#_process_mask), [`masks`](splash_attention_mask.md#MultiHeadMask.masks), [`_ComputableMask`](splash_attention_mask.md#_ComputableMask), [`__init__`](splash_attention_mask.md#LogicalAnd.__init__), [`__init__`](splash_attention_mask.md#LogicalOr.__init__), [`__post_init__`](splash_attention_mask.md#MultiHeadMask.__post_init__), [`MultiHeadMask`](splash_attention_mask.md#MultiHeadMask), [`make_attention_reference`](splash_attention_kernel.md#make_attention_reference), [`shape`](splash_attention_mask.md#MultiHeadMask.shape), [`left`](splash_attention_mask.md#LogicalAnd.left), [`left`](splash_attention_mask.md#LogicalOr.left), [`right`](splash_attention_mask.md#LogicalAnd.right), [`right`](splash_attention_mask.md#LogicalOr.right), [`_check_mask`](splash_attention_mask_info.md#_check_mask), [`shape`](splash_attention_mask.md#LogicalAnd.shape), [`shape`](splash_attention_mask.md#LogicalOr.shape), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`FullMask`](splash_attention_mask.md#FullMask), [`LogicalAnd`](splash_attention_mask.md#LogicalAnd), [`LogicalOr`](splash_attention_mask.md#LogicalOr)

### `MultiHeadMask`  ·  implements/extends Mask
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py:176`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L176)
- doc: Lazy multihead mask, combines multiple lazy masks one per head.
- signature: `class MultiHeadMask(Mask):`
- members:
  - `shape(self)` — [`L199`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L199)
  - `masks` — [`L179`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L179)
- protocol/private: `__eq__`[`L214`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L214), `__getitem__`[`L202`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L202), `__hash__`[`L220`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L220), `__post_init__`[`L181`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L181)
- uses (calls/refs, reference-scoped): [`shape`](splash_attention_mask.md#Mask.shape), [`Mask`](splash_attention_mask.md#Mask)
- used by: [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`_process_mask`](splash_attention_mask_info.md#_process_mask), [`shape`](splash_attention_mask.md#Mask.shape), [`Mask`](splash_attention_mask.md#Mask), [`_get_mask_info_for_shard`](splash_attention_mask_info.md#_get_mask_info_for_shard)

### `NumpyMask`  ·  implements/extends Mask
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py:489`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L489)
- doc: A mask backed by a dense numpy array.
- signature: `class NumpyMask(Mask):`
- members:
  - `shape(self)` — [`L502`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L502)
  - `array` — [`L492`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L492)
- protocol/private: `__eq__`[`L508`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L508), `__getitem__`[`L505`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L505), `__hash__`[`L514`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L514), `__post_init__`[`L494`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L494)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask)
- used by: [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`shape`](splash_attention_mask.md#Mask.shape), [`Mask`](splash_attention_mask.md#Mask)

### `_ComputableMask`  ·  implements/extends Mask
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py:224`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L224)
- doc: Superclass for all masks that can be computed inside the kernel using a callable object.
- signature: `class _ComputableMask(Mask):`
- members:
  - `shape(self)` — [`L268`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L268)
  - `mask_function` — [`L247`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L247)
  - `q_sequence` — [`L246`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L246)
- protocol/private: `__eq__`[`L287`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L287), `__getitem__`[`L271`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L271), `__hash__`[`L290`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L290), `__init__`[`L249`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L249), `_shape`[`L245`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L245)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`_fill_slice`](splash_attention_mask.md#_fill_slice), [`CausalMask`](splash_attention_mask.md#CausalMask), [`LocalMask`](splash_attention_mask.md#LocalMask), [`ChunkedCausalMask`](splash_attention_mask.md#ChunkedCausalMask)
- used by: [`shape`](splash_attention_mask.md#Mask.shape), [`Mask`](splash_attention_mask.md#Mask), [`__eq__`](splash_attention_mask.md#LocalMask.__eq__), [`__hash__`](splash_attention_mask.md#LocalMask.__hash__), [`__init__`](splash_attention_mask.md#LocalMask.__init__), [`__eq__`](splash_attention_mask.md#CausalMask.__eq__), [`__eq__`](splash_attention_mask.md#ChunkedCausalMask.__eq__), [`__hash__`](splash_attention_mask.md#CausalMask.__hash__), [`__hash__`](splash_attention_mask.md#ChunkedCausalMask.__hash__), [`__init__`](splash_attention_mask.md#CausalMask.__init__), [`__init__`](splash_attention_mask.md#ChunkedCausalMask.__init__), [`CausalMask`](splash_attention_mask.md#CausalMask), [`LocalMask`](splash_attention_mask.md#LocalMask), [`ChunkedCausalMask`](splash_attention_mask.md#ChunkedCausalMask)

## Functions
- `_fill_slice(inp_slice: slice, size: int)` — [`L518`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L518)
- `make_causal_mask(shape: tuple[int, int], offset: int = 0)` — [`L56`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L56) — Makes a causal attention mask.
- `make_chunk_attention_mask(shape: tuple[int, int], chunk_size: int)` — [`L94`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L94) — Makes a chunked causal attention mask.
- `make_local_attention_mask(shape: tuple[int, int], window_size: tuple[int | None, int | None], *, offset: int = 0)` — [`L75`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L75) — Makes a local attention mask.
- `make_random_mask(shape: tuple[int, int], sparsity: float, seed: int)` — [`L123`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py#L123) — Makes a random attention mask.

