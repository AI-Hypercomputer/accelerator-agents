---
title: 'Module: python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.update_kv_cache.update_kv_cache`/
symbols:
  kv_cache_update_impl: kv_cache_update_impl().
  kv_cache_update._kv_cache_update_wrapper: kv_cache_update()._kv_cache_update_wrapper().
  get_num_slices_per_block: get_num_slices_per_block().
  kv_cache_update: kv_cache_update().
  get_slot_mapping: get_slot_mapping().
  VMEM_SIZE: VMEM_SIZE.
  kv_cache_update_kernel: kv_cache_update_kernel().
---
# Module: [`python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py)

## Functions
- `_kv_cache_update_wrapper(new_kv, slices, kv_cache, num_kv_update_slices)` — [`L227`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py#L227)
- `get_num_slices_per_block(new_kv: jax.Array, kv_cache: jax.Array, page_size=128)` — [`L38`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py#L38) — new_kv: 5D [tokens, 1, heads*2//packing, packing, head_dim]
- `get_slot_mapping(num_slices_per_block: int, kv_cache_start_loc: jax.Array, new_kv_start_loc: jax.Array, slice_lens: jax.Array)` — [`L14`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py#L14)
- `kv_cache_update(new_kv: jax.Array, slices: jax.Array, kv_cache: jax.Array, num_kv_update_slices: jax.Array, *, page_size: int = 1, num_slices_per_block: int = 8, kv_partition_axis: str = "tensor")` — [`L201`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py#L201)
- `kv_cache_update_impl(new_kv, slices, kv_cache, num_kv_update_slices, page_size, num_slices_per_block)` — [`L120`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py#L120) — Accept 5D inputs. Flattens to 3D internally for Pallas kernel, reshapes output back to 5D.
- `kv_cache_update_kernel(slices_ref, new_kv_hbm_ref, kv_cache_hbm_ref, _, scratch, sem)` — [`L69`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py#L69)

## Module values
- `VMEM_SIZE` — [`L35`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py#L35)

