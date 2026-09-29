---
title: 'Module: src/maxtext/kernels/gather_reduce_pallas.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/gather_reduce_pallas.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.gather_reduce_pallas`/sc_gather_reduce().
symbols:
  sc_gather_reduce.kernel: kernel().
  sc_gather_reduce.kernel.idx_pipeline: kernel().idx_pipeline().
  sc_gather_reduce: ''
  sc_gather_reduce.kernel.idx_pipeline.data_pipeline: kernel().idx_pipeline().data_pipeline().
---
# Module: [`src/maxtext/kernels/gather_reduce_pallas.py`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_pallas.py)

## Functions
- `data_pipeline(gather_ref, out_ref)` — [`L146`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_pallas.py#L146)
- `idx_pipeline(idx_ref, weights_ref=None)` — [`L118`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_pallas.py#L118)
- `kernel(in_hbm_ref, idx_hbm_ref, weights_hbm_ref, out_hbm_ref)` — [`L100`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_pallas.py#L100)
- `sc_gather_reduce(op: jax.Array, idx: jax.Array, topk_weights: jax.Array | None = None, *, reduce_group_size: int, single_sc: bool = False, col_chunk_size: int = int(3.5 * 1024), row_chunk_size: int = 512, topk_wgt_zero_nan: bool = False)` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_pallas.py#L32) — Performs a gather-reduce operation on SparseCore.

