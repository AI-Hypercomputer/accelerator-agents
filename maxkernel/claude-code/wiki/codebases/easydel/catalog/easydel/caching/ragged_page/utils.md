---
title: 'Module: easydel/caching/ragged_page/utils.py'
type: catalog
provenance: extracted
module: easydel/caching/ragged_page/utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching.ragged_page.utils`/
symbols:
  kv_cache_update: kv_cache_update().
  kv_cache_update_jax: kv_cache_update_jax().
  localize_slice_indices_for_page_shard: localize_slice_indices_for_page_shard().
  cdiv: cdiv().
  _kv_cache_update_kernel: _kv_cache_update_kernel().
---
# Module: [`easydel/caching/ragged_page/utils.py`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/utils.py)

## Functions
- `_kv_cache_update_kernel(slice_indices_ref, new_kv_tokens_hbm_ref, kv_cache_pages_hbm_ref, _, vmem_scratch_buffer, dma_semaphore)` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/utils.py#L126) — Low-level TPU kernel for paged KV cache updates.
- `cdiv(a: int, v: int)` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/utils.py#L46) — Ceiling division: divide a by v and round up.
- `kv_cache_update(new_kv_tokens: Float[Array, "total_tokens num_combined_kv_heads head_dim"], slice_indices: Int[Array, "3 num_slices"], kv_cache_pages: Float[Array, "total_cache_positions num_combined_kv_heads head_dim"], total_update_slices: Int[Array, ""], *, page_size: int = 32, slices_per_processing_page: int = 8, page_shard_index: Int[Array, ""] | int = 0)` — [`L199`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/utils.py#L199) — TPU-optimized paged KV cache update using Pallas kernels.
- `kv_cache_update_jax(new_kv_tokens: Float[Array, "total_tokens num_kv_heads head_dim"], slice_indices: Int[Array, "3 num_slices"], kv_cache_pages: Float[Array, "total_cache_positions num_kv_heads head_dim"], total_update_slices: Int[Array, ""], *, page_size: int = 32, page_shard_index: Int[Array, ""] | int = 0)` — [`L303`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/utils.py#L303) — Portable JAX implementation of paged KV-cache update.
- `localize_slice_indices_for_page_shard(slice_indices: Int[Array, "3 num_slices"], total_update_slices: Int[Array, ""], *, page_size: int, local_flat_cache_positions: int, page_shard_index: Int[Array, ""] | int = 0)` — [`L66`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/utils.py#L66) — Translate global v2 slice mapping offsets to one page shard.

