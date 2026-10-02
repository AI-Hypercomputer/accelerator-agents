---
title: 'Module: tpu_inference/kernels/sparse_core/ragged_gather.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/sparse_core/ragged_gather.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.sparse_core.ragged_gather`/
symbols:
  ragged_gather: ragged_gather().
  main_kernel: main_kernel().
  main_kernel.inner_kernel: main_kernel().inner_kernel().
  calculate_col_size: calculate_col_size().
  main_kernel.inner_kernel.dma_write_loop: main_kernel().inner_kernel().dma_write_loop().
---
# Module: [`tpu_inference/kernels/sparse_core/ragged_gather.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather.py)

## Functions
- `calculate_col_size(hidden_size: int)` — [`L215`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather.py#L215) — Calculate col size for ragged gather kernel.
- `dma_write_loop(col_vmem_start)` — [`L137`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather.py#L137)
- `inner_kernel()` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather.py#L85)
- `main_kernel(start_ref: jax.Ref, end_ref: jax.Ref, in_hbm_ref: jax.Ref, indices_hbm_ref: jax.Ref, out_hbm_ref: jax.Ref, start_vmem_ref: jax.Ref, end_vmem_ref: jax.Ref, out_vmem_ref: jax.Ref, indices_vmem_ref: jax.Ref, sem_ref: jax.Ref, *, core_axis_name: str, subcore_axis_name: str)` — [`L26`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather.py#L26)
- `ragged_gather(x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array)` — [`L240`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather.py#L240) — Perform gather on indices within dynamic array start and end.

