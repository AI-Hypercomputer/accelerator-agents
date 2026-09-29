---
title: 'Module: axlearn/common/kv_cache/paged_kv_storage.py'
type: catalog
provenance: extracted
module: axlearn/common/kv_cache/paged_kv_storage.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kv_cache.paged_kv_storage`/
symbols:
  Bf16PagedStorage.as_dense: Bf16PagedStorage#as_dense().
  reconstruct_kv: reconstruct_kv().
  scatter_update_pages_kernel: scatter_update_pages_kernel().
  Bf16PagedStorage.kernel_inputs: Bf16PagedStorage#kernel_inputs().
  Bf16PagedStorage.page_indices: Bf16PagedStorage#page_indices.
  scatter_update_pages: scatter_update_pages().
  Bf16PagedStorage.k_proj: Bf16PagedStorage#k_proj.
  Bf16PagedStorage.v_proj: Bf16PagedStorage#v_proj.
  PagedKVStorage.as_dense: PagedKVStorage#as_dense().
  register_bf16_kernel.decorator: register_bf16_kernel().decorator().
  PagedKVStorage: PagedKVStorage#
  _BF16_KERNEL_REGISTRY._BF16_KERNEL_REGISTRY: _BF16_KERNEL_REGISTRY._BF16_KERNEL_REGISTRY.
  PagedKVStorage.page_indices: PagedKVStorage#page_indices.
  PagedKVStorage.key_positions: PagedKVStorage#key_positions.
  register_bf16_kernel: register_bf16_kernel().
  Bf16PagedStorage.key_positions: Bf16PagedStorage#key_positions.
  Bf16PagedStorage.kernel_for: Bf16PagedStorage#kernel_for().
  PagedKVStorage.kernel_inputs: PagedKVStorage#kernel_inputs().
  PagedKVStorage.kernel_for: PagedKVStorage#kernel_for().
  Bf16PagedStorage: Bf16PagedStorage#
---
# Module: [`axlearn/common/kv_cache/paged_kv_storage.py`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py)

## Classes
### `Bf16PagedStorage`  ·  implements/extends NamedTuple
- def: [`axlearn/common/kv_cache/paged_kv_storage.py:185`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L185)
- doc: Paged KV storage emitted by `PagedKVCache` (bf16 pages).
- signature: `class Bf16PagedStorage(NamedTuple):`
- members:
  - `as_dense(self)` — [`L232`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L232)
  - `kernel_for(cls, backend: str)` — [`L239`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L239)
  - `kernel_inputs(self, mha_dim_to_partition_spec: Mapping[str, PartitionSpec])` — [`L214`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L214)
  - `k_proj` — [`L209`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L209)
  - `key_positions` — [`L212`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L212)
  - `page_indices` — [`L211`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L211)
  - `v_proj` — [`L210`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L210)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`reconstruct_kv`](paged_kv_storage.md#reconstruct_kv), [`_BF16_KERNEL_REGISTRY`](paged_kv_storage.md#_BF16_KERNEL_REGISTRY._BF16_KERNEL_REGISTRY)

### `PagedKVStorage`  ·  implements/extends Protocol
- def: [`axlearn/common/kv_cache/paged_kv_storage.py:118`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L118)
- doc: Structural protocol for a paged KV storage variant.
- signature: `class PagedKVStorage(Protocol):`
- members:
  - `as_dense(self)` — [`L150`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L150) — Materialise dense `(k, v)` for fallback / debug paths.
  - `kernel_for(cls, backend: str)` — [`L158`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L158) — Return the attention kernel registered for this storage + backend.
  - `kernel_inputs(self, mha_dim_to_partition_spec: Mapping[str, PartitionSpec])` — [`L131`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L131) — Produce the paged-specific entries of the kernel's input dict.
  - `key_positions` — [`L129`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L129)
  - `page_indices` — [`L128`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L128)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor)
- used by: [`as_dense_kv`](paged_kv_cache.md#PagedKVCache.as_dense_kv)

## Functions
- `decorator(fn: Callable)` — [`L178`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L178)
- `reconstruct_kv(page_tables: Tensor, pages: Tensor)` — [`L37`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L37) — Retrieve key/value from page tables given pages.
- `register_bf16_kernel(backend: str)` — [`L165`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L165) — Register `fn` as the bf16 paged attention kernel for `backend`.
- `scatter_update_pages(kv_pages: Tensor, kv_proj: Tensor, page_indices: Tensor, key_positions: Tensor)` — [`L53`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L53) — Scatter kv_proj into kv_pages according to key_positions.
- `scatter_update_pages_kernel(*, kv_pages: Tensor, kv_proj: Tensor, page_indices: Tensor, key_positions: Tensor, shmap_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor])` — [`L78`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L78) — Equivalent to `scatter_update_pages` but much faster on TPU and GPU.

## Module values
- `_BF16_KERNEL_REGISTRY` — [`L162`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage.py#L162)

