---
title: 'Module: tpu_inference/kernels/sparse_core/gather_reduce.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/sparse_core/gather_reduce.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.sparse_core.gather_reduce`/
symbols:
  get_valid_col_chunk_size: get_valid_col_chunk_size().
  VectorTypeHelper: VectorTypeHelper#
  VectorTypeHelper.__init__: VectorTypeHelper#__init__().
  VectorTypeHelper.__getitem__: VectorTypeHelper#__getitem__().
  _I32: _I32.
  _F32: _F32.
  _BF16: _BF16.
  is_supported_by_sc_gather_reduce: is_supported_by_sc_gather_reduce().
  sc_gather_reduce: sc_gather_reduce().
---
# Module: [`tpu_inference/kernels/sparse_core/gather_reduce.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py)

## Classes
### `VectorTypeHelper`
- def: [`tpu_inference/kernels/sparse_core/gather_reduce.py:47`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py#L47)
- doc: Helper to create VectorType with a specific element type.
- signature: `class VectorTypeHelper:`
- protocol/private: `__getitem__`[`L53`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py#L53), `__init__`[`L50`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py#L50)

## Functions
- `get_valid_col_chunk_size(hidden_size: int)` — [`L37`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py#L37)
- `is_supported_by_sc_gather_reduce(x_shape: int, sc_kernel_threshold: int)` — [`L64`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py#L64)
- `sc_gather_reduce(op: jax.Array, idx: jax.Array, topk_weights: jax.Array | None=None, *, reduce_group_size: int, single_sc: bool=False, col_chunk_size: int=int(3.5 * 1024), row_chunk_size: int=16, loop_unroll_factor_1: int=2, loop_unroll_factor_2: int=2, loop_unroll_factor_3: int=8, loop_parallel_access_1: bool=True, loop_parallel_access_2: bool=False, loop_parallel_access_3: bool=False, topk_wgt_zero_nan: bool=False)` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py#L85) — Performs a gather-reduce operation on SparseCore.

## Module values
- `_BF16` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py#L61)
- `_F32` — [`L60`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py#L60)
- `_I32` — [`L59`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py#L59)

