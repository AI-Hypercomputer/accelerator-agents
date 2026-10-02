---
title: 'Module: tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_gather.pallas_mosaic_tpu_kernel`/
symbols:
  ragged_gather_pallas: ragged_gather_pallas().
  main_kernel: main_kernel().
  main_kernel.inner_kernel: main_kernel().inner_kernel().
  calculate_col_size: calculate_col_size().
  main_kernel.inner_kernel.dma_write_loop: main_kernel().inner_kernel().dma_write_loop().
---
# Module: [`tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py)

## Functions
- `calculate_col_size(hidden_size: int)` — [`L211`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py#L211) — Calculate col size for ragged gather kernel.
- `dma_write_loop(col_vmem_start)` — [`L136`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py#L136)
- `inner_kernel()` — [`L86`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py#L86)
- `main_kernel(start_ref: jax.Ref, end_ref: jax.Ref, in_hbm_ref: jax.Ref, indices_hbm_ref: jax.Ref, out_hbm_ref: jax.Ref, start_vmem_ref: jax.Ref, end_vmem_ref: jax.Ref, out_vmem_ref: jax.Ref, indices_vmem_ref: jax.Ref, sem_ref: jax.Ref, *, core_axis_name: str, subcore_axis_name: str)` — [`L26`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py#L26)
- `ragged_gather_pallas(x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array)` — [`L236`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py#L236) — Perform gather on indices within dynamic array start and end.

