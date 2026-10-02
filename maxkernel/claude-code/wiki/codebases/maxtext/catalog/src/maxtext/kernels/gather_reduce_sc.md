---
title: 'Module: src/maxtext/kernels/gather_reduce_sc.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/gather_reduce_sc.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.gather_reduce_sc`/
symbols:
  sc_gather_reduce._kernel_impl.const_lut: sc_gather_reduce()._kernel_impl().const_lut().
  _I32: _I32.
  sc_gather_reduce._kernel_impl: sc_gather_reduce()._kernel_impl().
  sc_gather_reduce._kernel_impl.perform_add: sc_gather_reduce()._kernel_impl().perform_add().
  sc_gather_reduce._kernel_impl.perform_add.get_row_val: sc_gather_reduce()._kernel_impl().perform_add().get_row_val().
  sc_gather_reduce._kernel_impl.fill_out_offset_tile: sc_gather_reduce()._kernel_impl().fill_out_offset_tile().
  _F32: _F32.
  sc_gather_reduce._kernel_impl.fill_load_offset_tile: sc_gather_reduce()._kernel_impl().fill_load_offset_tile().
  sc_gather_reduce._kernel_impl.load_weights: sc_gather_reduce()._kernel_impl().load_weights().
  _BF16: _BF16.
  sc_gather_reduce: sc_gather_reduce().
  sc_gather_reduce.kernel_main: sc_gather_reduce().kernel_main().
  VectorTypeHelper: VectorTypeHelper#
  VectorTypeHelper.__getitem__: VectorTypeHelper#__getitem__().
  VectorTypeHelper.element_type_fn: VectorTypeHelper#element_type_fn.
  VectorTypeHelper.__init__: VectorTypeHelper#__init__().
---
# Module: [`src/maxtext/kernels/gather_reduce_sc.py`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py)

## Classes
### `VectorTypeHelper`
- def: [`src/maxtext/kernels/gather_reduce_sc.py:41`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L41)
- doc: Helper to create VectorType with a specific element type.
- signature: `class VectorTypeHelper:`
- members:
  - `element_type_fn` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L45)
- protocol/private: `__getitem__`[`L47`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L47), `__init__`[`L44`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L44)
- used by: [`_I32`](gather_reduce_sc.md#_I32), [`_F32`](gather_reduce_sc.md#_F32), [`_BF16`](gather_reduce_sc.md#_BF16)

## Functions
- `_kernel_impl(current_sc_core, current_local_core, idx_ref, op_ref, weights_ref, out_ref, func_op)` — [`L178`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L178)
- `const_lut(val, ty=None)` — [`L189`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L189)
- `fill_load_offset_tile(offset_tile_local, idx_tile_local, col_pos)` — [`L196`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L196) — Fills the offset tile for indirect DMA gather.
- `fill_out_offset_tile(offset_tile_out_local, col_pos, row_pos=None)` — [`L664`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L664) — Fills the offset tile for indirect DMA scatter for outputs (bf16).
- `get_row_val(row_idx)` — [`L529`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L529)
- `kernel_main(current_sc_core, current_local_core, idx_ref, op_ref, weights_ref, out_ref, func_op)` — [`L1514`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L1514)
- `load_weights(lin_idx, dst_tile, sflag)` — [`L343`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L343) — Loads weights from HBM to TileSpMem.
- `perform_add(scratch_local, scratch_out_local, idx_parity, weights_local=None, parity=None)` — [`L399`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L399) — Performs reduction (summation) of rows in scratchpad.
- `sc_gather_reduce(op: jax.Array, idx: jax.Array, topk_weights: jax.Array | None = None, *, reduce_group_size: int, single_sc: bool = False, col_chunk_size: int = int(3.5 * 1024), row_chunk_size: int = 16, loop_unroll_factor_1: int = 2, loop_unroll_factor_2: int = 2, loop_unroll_factor_3: int = 8, loop_parallel_access_1: bool = True, loop_parallel_access_2: bool = False, loop_parallel_access_3: bool = False, topk_wgt_zero_nan: bool = False)` — [`L72`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L72) — Performs a gather-reduce operation on SparseCore.

## Module values
- `_BF16` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L55)
- `_F32` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L54)
- `_I32` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L53)

