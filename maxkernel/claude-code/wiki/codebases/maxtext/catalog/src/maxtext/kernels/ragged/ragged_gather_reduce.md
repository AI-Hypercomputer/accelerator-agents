---
title: 'Module: src/maxtext/kernels/ragged/ragged_gather_reduce.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/ragged/ragged_gather_reduce.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.ragged.ragged_gather_reduce`/
symbols:
  ragged_gather_reduce: ragged_gather_reduce().
  _COMPILER_PARAMS: _COMPILER_PARAMS.
  main_kernel: main_kernel().
  _OUT_KW: _OUT_KW.
  _SCRATCH_KW: _SCRATCH_KW.
  _align_to: _align_to().
  get_cost_estimate: get_cost_estimate().
  _fallback_implementation: _fallback_implementation().
  main_kernel.inner_kernel: main_kernel().inner_kernel().
  _preprocess: _preprocess().
  main_kernel.inner_kernel.row_loop: main_kernel().inner_kernel().row_loop().
  main_kernel.inner_kernel.row_loop.dma_write_loop: main_kernel().inner_kernel().row_loop().dma_write_loop().
---
# Module: [`src/maxtext/kernels/ragged/ragged_gather_reduce.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py)

## Functions
- `_align_to(a, b)` — [`L48`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L48)
- `_fallback_implementation(x: jax.Array, indices: jax.Array, topk_weights: jax.Array, valid_rows_mask: jax.Array, reduce_group_size: int)` — [`L107`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L107) — Fallback to JAX implementation.
- `_preprocess(indices: jax.Array, topk_weights: jax.Array, valid_rows_mask: jax.Array, reduce_group_size: int, num_row_partitions: int, num_simd_lanes: int)` — [`L381`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L381) — Preprocesses indices for ragged gather reduce.
- `dma_write_loop(col_vmem_start, carry)` — [`L251`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L251)
- `get_cost_estimate(padded_input_size: int, aligned_hidden_size: int, reduce_group_size: int, input_dtype_bytes: int, bytes_accessed_override: int = -1, flops_override: int = -1)` — [`L52`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L52) — Returns a cost estimate for the ragged gather-reduce kernel.
- `inner_kernel()` — [`L163`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L163)
- `main_kernel(num_rows_per_row_partition_ref: jax.Ref, in_hbm_ref: jax.Ref, src_indices_hbm_ref: jax.Ref, dst_indices_hbm_ref: jax.Ref, topk_weights_hbm_ref: jax.Ref, out_hbm_ref: jax.Ref, num_rows_per_row_partition_vmem_ref: jax.Ref, out_vmem_ref: jax.Ref, prev_iter_last_row_vmem_ref: jax.Ref, src_indices_vmem_ref: jax.Ref, dst_indices_vmem_ref: jax.Ref, topk_weights_vmem_ref: jax.Ref, sem_ref: jax.Ref, *, core_axis_name: str, subcore_axis_name: str, num_row_partitions: int, num_column_partitions: int)` — [`L122`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L122) — Main Pallas kernel for ragged gather and reduction on SparseCore.
- `ragged_gather_reduce(x: jax.Array, indices: jax.Array, topk_weights: jax.Array, valid_rows_mask: jax.Array, reduce_group_size: int, enforce_fallback: bool = False, flops_override: int = -1, bytes_accessed_override: int = -1)` — [`L435`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L435) — Gathers `x` according to `indices`, applies weights and masks, and reduces. — documented in [maxtext-layers-moe](../../../../../concepts/maxtext-layers-moe.md)
- `row_loop(row_block_id)` — [`L192`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L192)

## Module values
- `_COMPILER_PARAMS` — [`L33`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L33)
- `_OUT_KW` — [`L31`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L31)
- `_SCRATCH_KW` — [`L32`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L32)

