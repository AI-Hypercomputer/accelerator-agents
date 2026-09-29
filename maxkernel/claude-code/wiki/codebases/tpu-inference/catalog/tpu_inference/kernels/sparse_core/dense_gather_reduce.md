---
title: 'Module: tpu_inference/kernels/sparse_core/dense_gather_reduce.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/sparse_core/dense_gather_reduce.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.sparse_core.dense_gather_reduce`/
symbols:
  dense_gather_reduce: dense_gather_reduce().
  _sc_gather_reduce: _sc_gather_reduce().
  _sc_gather_reduce.kernel: _sc_gather_reduce().kernel().
  _sc_gather_reduce.kernel.idx_pipeline: _sc_gather_reduce().kernel().idx_pipeline().
  is_compatible: is_compatible().
  _sc_gather_reduce.kernel.idx_pipeline.data_pipeline: _sc_gather_reduce().kernel().idx_pipeline().data_pipeline().
  _jax_fallback: _jax_fallback().
---
# Module: [`tpu_inference/kernels/sparse_core/dense_gather_reduce.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/dense_gather_reduce.py)

## Functions
- `_jax_fallback(x, indices, topk_weights, reduce_group_size, topk_wgt_zero_nan=False)` — [`L244`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/dense_gather_reduce.py#L244)
- `_sc_gather_reduce(op: jax.Array, idx: jax.Array, topk_weights: jax.Array | None = None, *, reduce_group_size: int, single_sc: bool = False, col_chunk_size: int = int(3.5 * 1024), row_chunk_size: int = 512, topk_wgt_zero_nan: bool = False)` — [`L66`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/dense_gather_reduce.py#L66) — Performs a gather-reduce operation on SparseCore.
- `data_pipeline(gather_ref, out_ref)` — [`L186`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/dense_gather_reduce.py#L186)
- `dense_gather_reduce(x: jax.Array, indices: jax.Array, topk_weights: jax.Array, reduce_group_size: int, topk_wgt_zero_nan: bool = False)` — [`L271`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/dense_gather_reduce.py#L271) — Wrapper that redirects to Pallas dense gather reduce kernel if constraints are met.
- `idx_pipeline(idx_ref, weights_ref=None)` — [`L155`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/dense_gather_reduce.py#L155)
- `is_compatible(op: jax.Array, idx: jax.Array, reduce_group_size: int, row_chunk_size: int = 512, single_sc: bool = False)` — [`L31`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/dense_gather_reduce.py#L31) — Checks if the inputs are compatible with the SparseCore Pallas kernel.
- `kernel(in_hbm_ref, idx_hbm_ref, weights_hbm_ref, out_hbm_ref)` — [`L133`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/dense_gather_reduce.py#L133)

