---
title: 'Module: easydel/caching/_specs.py'
type: catalog
provenance: extracted
module: easydel/caching/_specs.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching._specs`/
symbols:
  AttentionSpec.page_size_bytes: AttentionSpec#page_size_bytes().
  AttentionSpec: AttentionSpec#
  KVCacheSpec.type_id: KVCacheSpec#type_id().
  KVCacheSpec.max_memory_usage_bytes: KVCacheSpec#max_memory_usage_bytes().
  FullAttentionSpec.merge: FullAttentionSpec#merge().
  ChunkedLocalAttentionSpec.max_memory_usage_bytes: ChunkedLocalAttentionSpec#max_memory_usage_bytes().
  SlidingWindowSpec.max_memory_usage_bytes: SlidingWindowSpec#max_memory_usage_bytes().
  MambaSpec.page_size_bytes: MambaSpec#page_size_bytes().
  KVCacheSpec: KVCacheSpec#
  FullAttentionSpec.max_memory_usage_bytes: FullAttentionSpec#max_memory_usage_bytes().
  ChunkedLocalAttentionSpec.type_id: ChunkedLocalAttentionSpec#type_id().
  SlidingWindowSpec.type_id: SlidingWindowSpec#type_id().
  KVCacheSpec.page_size: KVCacheSpec#page_size.
  KVCacheSpec.merge: KVCacheSpec#merge().
  KVCacheSpec.page_size_bytes: KVCacheSpec#page_size_bytes().
  FullAttentionSpec.type_id: FullAttentionSpec#type_id().
  MambaSpec.type_id: MambaSpec#type_id().
  FullAttentionSpec.sliding_window: FullAttentionSpec#sliding_window.
  FullAttentionSpec.attention_chunk_size: FullAttentionSpec#attention_chunk_size.
  MambaSpec.page_size_padded: MambaSpec#page_size_padded.
  FullAttentionSpec: FullAttentionSpec#
  ChunkedLocalAttentionSpec: ChunkedLocalAttentionSpec#
  SlidingWindowSpec: SlidingWindowSpec#
  MambaSpec: MambaSpec#
  MambaSpec.num_elements: MambaSpec#num_elements.
  cdiv: cdiv().
  SlidingWindowSpec.__post_init__: SlidingWindowSpec#__post_init__().
  MambaSpec.max_memory_usage_bytes: MambaSpec#max_memory_usage_bytes().
  AttentionSpec.use_mla: AttentionSpec#use_mla.
  FullAttentionSpec.merge_window_sizes: FullAttentionSpec#merge_window_sizes().
  ChunkedLocalAttentionSpec.attention_chunk_size: ChunkedLocalAttentionSpec#attention_chunk_size.
  SlidingWindowSpec.sliding_window: SlidingWindowSpec#sliding_window.
  MambaSpec.shapes: MambaSpec#shapes.
  MambaSpec.dtype: MambaSpec#dtype.
  AttentionSpec.num_kv_heads: AttentionSpec#num_kv_heads.
  AttentionSpec.head_size: AttentionSpec#head_size.
  AttentionSpec.dtype: AttentionSpec#dtype.
  MambaSpec.__post_init__: MambaSpec#__post_init__().
---
# Module: [`easydel/caching/_specs.py`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py)

## Classes
### `AttentionSpec`  ·  implements/extends KVCacheSpec
- def: [`easydel/caching/_specs.py:223`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L223)
- doc: Base specification for attention-based cache formats.
- signature: `class AttentionSpec(KVCacheSpec):`
- members:
  - `page_size_bytes(self)` — [`L246`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L246) — Calculate page size for attention cache in bytes.
  - `dtype` — [`L242`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L242)
  - `head_size` — [`L241`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L241)
  - `num_kv_heads` — [`L240`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L240)
  - `use_mla` — [`L243`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L243)
- uses (calls/refs, reference-scoped): [`KVCacheSpec`](_specs.md#KVCacheSpec), [`page_size`](_specs.md#KVCacheSpec.page_size), [`ChunkedLocalAttentionSpec`](_specs.md#ChunkedLocalAttentionSpec), [`FullAttentionSpec`](_specs.md#FullAttentionSpec), [`SlidingWindowSpec`](_specs.md#SlidingWindowSpec)
- used by: [`max_memory_usage_bytes`](_specs.md#ChunkedLocalAttentionSpec.max_memory_usage_bytes), [`max_memory_usage_bytes`](_specs.md#SlidingWindowSpec.max_memory_usage_bytes), [`KVCacheSpec`](_specs.md#KVCacheSpec), [`max_memory_usage_bytes`](_specs.md#FullAttentionSpec.max_memory_usage_bytes), [`type_id`](_specs.md#ChunkedLocalAttentionSpec.type_id), [`type_id`](_specs.md#SlidingWindowSpec.type_id), [`page_size_bytes`](_specs.md#KVCacheSpec.page_size_bytes), [`type_id`](_specs.md#FullAttentionSpec.type_id), [`ChunkedLocalAttentionSpec`](_specs.md#ChunkedLocalAttentionSpec), [`FullAttentionSpec`](_specs.md#FullAttentionSpec), [`SlidingWindowSpec`](_specs.md#SlidingWindowSpec), [`__post_init__`](_specs.md#SlidingWindowSpec.__post_init__)

### `ChunkedLocalAttentionSpec`  ·  implements/extends AttentionSpec
- def: [`easydel/caching/_specs.py:355`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L355)
- doc: Specification for chunked local attention caching.
- signature: `class ChunkedLocalAttentionSpec(AttentionSpec):`
- members:
  - `max_memory_usage_bytes(self, max_model_len: int, max_num_batched_tokens: int, **kwargs)` — [`L376`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L376) — Calculate maximum memory for chunked attention cache.
  - `type_id(self)` — [`L373`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L373)
  - `attention_chunk_size` — [`L370`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L370)
- uses (calls/refs, reference-scoped): [`page_size_bytes`](_specs.md#AttentionSpec.page_size_bytes), [`AttentionSpec`](_specs.md#AttentionSpec), [`page_size`](_specs.md#KVCacheSpec.page_size), [`cdiv`](_specs.md#cdiv)
- used by: [`AttentionSpec`](_specs.md#AttentionSpec), [`type_id`](_specs.md#KVCacheSpec.type_id), [`max_memory_usage_bytes`](_specs.md#KVCacheSpec.max_memory_usage_bytes)

### `FullAttentionSpec`  ·  implements/extends AttentionSpec
- def: [`easydel/caching/_specs.py:264`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L264)
- doc: Specification for full attention caching.
- signature: `class FullAttentionSpec(AttentionSpec):`
- members:
  - `max_memory_usage_bytes(self, max_model_len: int, **kwargs)` — [`L296`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L296) — Calculate maximum memory for full attention cache.
  - `merge(cls, specs: list[Self])` — [`L336`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L336) — Merge a list of FullAttentionSpec objects into a single
  - `merge_window_sizes(cls, window_sizes: set[int])` — [`L313`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L313) — Merge sliding window sizes from multiple layers.
  - `type_id(self)` — [`L293`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L293)
  - `attention_chunk_size` — [`L290`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L290)
  - `sliding_window` — [`L289`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L289)
- uses (calls/refs, reference-scoped): [`page_size_bytes`](_specs.md#AttentionSpec.page_size_bytes), [`AttentionSpec`](_specs.md#AttentionSpec), [`page_size`](_specs.md#KVCacheSpec.page_size), [`merge`](_specs.md#KVCacheSpec.merge), [`cdiv`](_specs.md#cdiv)
- used by: [`AttentionSpec`](_specs.md#AttentionSpec), [`type_id`](_specs.md#KVCacheSpec.type_id), [`max_memory_usage_bytes`](_specs.md#KVCacheSpec.max_memory_usage_bytes), [`merge`](_specs.md#KVCacheSpec.merge)

### `KVCacheSpec`
- def: [`easydel/caching/_specs.py:82`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L82)
- doc: Base specification for key-value cache formats.
- signature: `class KVCacheSpec:`
- members:
  - `max_memory_usage_bytes(self, *args, **kwargs)` — [`L163`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L163) — Calculate maximum memory required for this cache configuration.
  - `merge(cls, specs: list[Self])` — [`L190`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L190) — Merge multiple cache specifications into a single specification.
  - `page_size_bytes(self)` — [`L140`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L140) — Calculate the memory size of a single cache page in bytes.
  - `type_id(self)` — [`L112`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L112) — Unique identifier for this cache specification type.
  - `page_size` — [`L109`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L109)
- uses (calls/refs, reference-scoped): [`page_size_bytes`](_specs.md#AttentionSpec.page_size_bytes), [`AttentionSpec`](_specs.md#AttentionSpec), [`max_memory_usage_bytes`](_specs.md#ChunkedLocalAttentionSpec.max_memory_usage_bytes), [`max_memory_usage_bytes`](_specs.md#SlidingWindowSpec.max_memory_usage_bytes), [`merge`](_specs.md#FullAttentionSpec.merge), [`page_size_bytes`](_specs.md#MambaSpec.page_size_bytes), [`max_memory_usage_bytes`](_specs.md#FullAttentionSpec.max_memory_usage_bytes), [`type_id`](_specs.md#ChunkedLocalAttentionSpec.type_id), [`type_id`](_specs.md#SlidingWindowSpec.type_id), [`type_id`](_specs.md#FullAttentionSpec.type_id), [`type_id`](_specs.md#MambaSpec.type_id), [`MambaSpec`](_specs.md#MambaSpec), [`max_memory_usage_bytes`](_specs.md#MambaSpec.max_memory_usage_bytes)
- used by: [`page_size_bytes`](_specs.md#AttentionSpec.page_size_bytes), [`AttentionSpec`](_specs.md#AttentionSpec), [`max_memory_usage_bytes`](_specs.md#ChunkedLocalAttentionSpec.max_memory_usage_bytes), [`max_memory_usage_bytes`](_specs.md#SlidingWindowSpec.max_memory_usage_bytes), [`merge`](_specs.md#FullAttentionSpec.merge), [`max_memory_usage_bytes`](_specs.md#FullAttentionSpec.max_memory_usage_bytes), [`type_id`](_specs.md#ChunkedLocalAttentionSpec.type_id), [`type_id`](_specs.md#SlidingWindowSpec.type_id), [`type_id`](_specs.md#FullAttentionSpec.type_id), [`MambaSpec`](_specs.md#MambaSpec)

### `MambaSpec`  ·  implements/extends KVCacheSpec
- def: [`easydel/caching/_specs.py:459`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L459)
- doc: Specification for Mamba state-space model caching.
- signature: `class MambaSpec(KVCacheSpec):`
- members:
  - `__post_init__(self)` — [`L483`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L483) — Calculate total elements from shapes.
  - `max_memory_usage_bytes(self, *args, **kwargs)` — [`L518`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L518) — Calculate maximum memory for Mamba state cache.
  - `page_size_bytes(self)` — [`L495`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L495) — Calculate page size for Mamba state cache in bytes.
  - `type_id(self)` — [`L491`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L491)
  - `dtype` — [`L480`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L480)
  - `num_elements` — [`L488`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L488)
  - `page_size_padded` — [`L481`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L481)
  - `shapes` — [`L479`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L479)
- uses (calls/refs, reference-scoped): [`KVCacheSpec`](_specs.md#KVCacheSpec)
- used by: [`type_id`](_specs.md#KVCacheSpec.type_id), [`max_memory_usage_bytes`](_specs.md#KVCacheSpec.max_memory_usage_bytes), [`KVCacheSpec`](_specs.md#KVCacheSpec), [`page_size_bytes`](_specs.md#KVCacheSpec.page_size_bytes)

### `SlidingWindowSpec`  ·  implements/extends AttentionSpec
- def: [`easydel/caching/_specs.py:401`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L401)
- doc: Specification for sliding window attention caching.
- signature: `class SlidingWindowSpec(AttentionSpec):`
- members:
  - `__post_init__(self)` — [`L421`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L421) — Validate sliding window configuration.
  - `max_memory_usage_bytes(self, max_model_len: int, max_num_batched_tokens: int, **kwargs)` — [`L434`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L434) — Calculate maximum memory for sliding window cache.
  - `type_id(self)` — [`L431`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L431)
  - `sliding_window` — [`L419`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L419)
- uses (calls/refs, reference-scoped): [`page_size_bytes`](_specs.md#AttentionSpec.page_size_bytes), [`AttentionSpec`](_specs.md#AttentionSpec), [`page_size`](_specs.md#KVCacheSpec.page_size), [`cdiv`](_specs.md#cdiv), [`use_mla`](_specs.md#AttentionSpec.use_mla)
- used by: [`AttentionSpec`](_specs.md#AttentionSpec), [`type_id`](_specs.md#KVCacheSpec.type_id), [`max_memory_usage_bytes`](_specs.md#KVCacheSpec.max_memory_usage_bytes)

## Functions
- `cdiv(a: int, b: int)` — [`L58`](../../../../../../raw/code/EasyDeL/easydel/caching/_specs.py#L58) — Ceiling division: divide a by b and round up.

