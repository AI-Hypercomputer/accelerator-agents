---
title: 'Module: tpu_inference/kernels/sparse_core/ragged_gather_v2.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/sparse_core/ragged_gather_v2.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.sparse_core.ragged_gather_v2`/
symbols:
  ragged_gather_v2: ragged_gather_v2().
  main_kernel_v2: main_kernel_v2().
  main_kernel_v2.inner_pipeline: main_kernel_v2().inner_pipeline().
  main_kernel_v2.outer_pipeline: main_kernel_v2().outer_pipeline().
  calculate_col_size: calculate_col_size().
  main_kernel_v2.col_loop: main_kernel_v2().col_loop().
---
# Module: [`tpu_inference/kernels/sparse_core/ragged_gather_v2.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_v2.py)

## Functions
- `calculate_col_size(hidden_size: int, packing: int)` — [`L27`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_v2.py#L27) — Calculates the max column size bounded by VMEM limits and hidden_size divisibility.
- `col_loop(col_base, gather_ref, out_ref, idx_rem, unpack_col_chunk)` — [`L130`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_v2.py#L130)
- `inner_pipeline(gather_ref, out_ref, idx_ref, unpack_col_chunk)` — [`L155`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_v2.py#L155)
- `main_kernel_v2(start_ref: jax.Ref, end_ref: jax.Ref, in_hbm_ref: jax.Ref, indices_hbm_ref: jax.Ref, out_hbm_ref: jax.Ref, start_vmem_ref: jax.Ref, end_vmem_ref: jax.Ref, sem_ref: jax.Ref, *, core_axis_name: str, subcore_axis_name: str, num_row_subchunks: int)` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_v2.py#L56)
- `outer_pipeline(idx_ref)` — [`L175`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_v2.py#L175)
- `ragged_gather_v2(x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array)` — [`L219`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_v2.py#L219) — Perform gather on indices within dynamic array start and end using BlockSpec.

