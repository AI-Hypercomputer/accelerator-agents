---
title: 'Module: tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.ragged_paged_attention.v2.ragged_kv_cache_update`/
symbols:
  _ceil_div: _ceil_div().
  _kv_cache_update_kernel: _kv_cache_update_kernel().
  _dynamic_validate_inputs: _dynamic_validate_inputs().
  _kv_cache_update: _kv_cache_update().
  _prev_power_of_2: _prev_power_of_2().
  _get_page_size_bytes: _get_page_size_bytes().
  _get_num_slices_per_kv_cache_update_block: _get_num_slices_per_kv_cache_update_block().
  kv_cache_update: kv_cache_update().
---
# Module: [`tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py)

## Functions
- `_ceil_div(a, b)` — [`L16`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py#L16)
- `_dynamic_validate_inputs(slices, new_token_num, kv_cache_token_num, page_size, num_slices)` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py#L78)
- `_get_num_slices_per_kv_cache_update_block(page_size_bytes: int, vmem_limit_bytes: int)` — [`L214`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py#L214) — Find the optimum number of slices to copy per Pallas program instance.
- `_get_page_size_bytes(block_size: int, num_combined_kv_heads: int, head_size: int, kv_cache_dtype)` — [`L200`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py#L200) — Returns the size in bytes of one page of the KV cache.
- `_kv_cache_update(new_kv: jax.Array, slices: jax.Array, kv_cache: jax.Array, num_slices: jax.Array, page_size: int, num_slices_per_block: int, dynamic_validate_inputs: bool, vmem_limit_bytes: int=40 * 1024 * 1024)` — [`L135`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py#L135)
- `_kv_cache_update_kernel(slices_ref, num_slices_ref, new_kv_hbm_ref, kv_cache_hbm_ref, _, scratch, sem)` — [`L21`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py#L21)
- `_prev_power_of_2(n: int)` — [`L193`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py#L193) — The previous power of 2 (inclusive)
- `kv_cache_update(new_kv: jax.Array, slices: jax.Array, kv_cache: jax.Array, num_slices: jax.Array, *, page_size: int=32, num_slices_per_block: int | None=None, mesh: Mesh | None=None, kv_cache_pspec: P | None=None, dynamic_validate_inputs: bool=False, vmem_limit_bytes: int=40 * 1024 * 1024)` — [`L238`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py#L238)

