---
title: 'Module: tpu_inference/kernels/sparse_core/ragged_scatter.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/sparse_core/ragged_scatter.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.sparse_core.ragged_scatter`/
symbols:
  main_kernel: main_kernel().
  calculate_col_size: calculate_col_size().
  _preprocess_indices: _preprocess_indices().
  ragged_scatter: ragged_scatter().
---
# Module: [`tpu_inference/kernels/sparse_core/ragged_scatter.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_scatter.py)

## Functions
- `_preprocess_indices(indices: jax.Array, start: jax.Array, end: jax.Array, out_pad_size: int, packing: int, row_tile_size: int)` — [`L292`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_scatter.py#L292) — Preprocesses indices for ragged scatter kernel.
- `calculate_col_size(hidden_size: int)` — [`L268`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_scatter.py#L268) — Calculate col size for ragged gather kernel.
- `main_kernel(total_num_rows_ref: jax.Ref, in_hbm_ref: jax.Ref, src_indices_hbm_ref: jax.Ref, dst_indices_hbm_ref: jax.Ref, out_hbm_ref: jax.Ref, total_num_rows_vmem_ref: jax.Ref, out_vmem_ref: jax.Ref, src_indices_vmem_ref: jax.Ref, dst_indices_vmem_ref: jax.Ref, sem_ref: jax.Ref, *, core_axis_name: str, subcore_axis_name: str)` — [`L26`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_scatter.py#L26)
- `ragged_scatter(x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array)` — [`L354`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_scatter.py#L354) — Gathers rows from `x` according to `indices` within a specified range.

