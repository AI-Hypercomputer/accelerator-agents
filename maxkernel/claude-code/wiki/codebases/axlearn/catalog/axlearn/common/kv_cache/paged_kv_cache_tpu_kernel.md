---
title: 'Module: axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py'
type: catalog
provenance: extracted
module: axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kv_cache.paged_kv_cache_tpu_kernel`/
symbols:
  tpu_scatter_update_pages_shmap_fn: tpu_scatter_update_pages_shmap_fn().
  _scatter_pages_kernel: _scatter_pages_kernel().
  tpu_scatter_update_pages_shmap_fn.block_idx_map: tpu_scatter_update_pages_shmap_fn().block_idx_map().
  _scatter_pages_kernel.compute: _scatter_pages_kernel().compute().
  _scatter_pages_kernel.pass_through: _scatter_pages_kernel().pass_through().
---
# Module: [`axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py)

## Functions
- `_scatter_pages_kernel(page_indices: Tensor, key_positions: Tensor, kv_pages: Tensor, kv_proj: Tensor, out_kv_pages: Tensor, *, page_size: int)` — [`L22`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py#L22)
- `block_idx_map(i, batch, page_indices, key_positions)` — [`L73`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py#L73)
- `compute()` — [`L40`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py#L40)
- `pass_through()` — [`L58`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py#L58)
- `tpu_scatter_update_pages_shmap_fn(kv_pages: Tensor, kv_proj: Tensor, page_indices: Tensor, key_positions: Tensor)` — [`L62`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py#L62)

