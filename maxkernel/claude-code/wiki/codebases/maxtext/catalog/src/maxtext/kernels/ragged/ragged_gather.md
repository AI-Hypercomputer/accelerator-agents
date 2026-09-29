---
title: 'Module: src/maxtext/kernels/ragged/ragged_gather.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/ragged/ragged_gather.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.ragged.ragged_gather`/
symbols:
  ragged_gather: ragged_gather().
  _COMPILER_PARAMS: _COMPILER_PARAMS.
  main_kernel: main_kernel().
  _OUT_KW: _OUT_KW.
  _SCRATCH_KW: _SCRATCH_KW.
  main_kernel.inner_kernel: main_kernel().inner_kernel().
  get_cost_estimate: get_cost_estimate().
  _fallback_implementation: _fallback_implementation().
  calculate_col_size: calculate_col_size().
  main_kernel.inner_kernel.dma_write_loop: main_kernel().inner_kernel().dma_write_loop().
---
# Module: [`src/maxtext/kernels/ragged/ragged_gather.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py)

## Functions
- `_fallback_implementation(x: jax.Array, indices: jax.Array, weights: jax.Array | None = None, has_weights: bool = False)` — [`L327`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L327) — Fallback to (non-ragged) JAX implementation for ragged gather.
- `calculate_col_size(hidden_size: int)` — [`L340`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L340) — Calculate col size for ragged gather kernel.
- `dma_write_loop(col_vmem_start)` — [`L160`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L160)
- `get_cost_estimate(out_size: int, hidden_size: int, dtype_bytes: int, has_weights: bool, flops_override: int = -1, bytes_accessed_override: int = -1)` — [`L274`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L274) — Returns a cost estimate for the ragged gather kernel.
- `inner_kernel()` — [`L108`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L108)
- `main_kernel(start_ref: jax.Ref, end_ref: jax.Ref, in_hbm_ref: jax.Ref, indices_hbm_ref: jax.Ref, weights_hbm_ref: jax.Ref, out_hbm_ref: jax.Ref, start_vmem_ref: jax.Ref, end_vmem_ref: jax.Ref, out_vmem_ref: jax.Ref, indices_vmem_ref: jax.Ref, weights_vmem_ref: jax.Ref, sem_ref: jax.Ref, *, core_axis_name: str, subcore_axis_name: str, has_weights: bool)` — [`L45`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L45) — Core ragged gather operation with per-row weighting.
- `ragged_gather(x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array, weights: jax.Array | None = None, has_weights: bool = False, enforce_fallback: bool = False, flops_override: int = -1, bytes_accessed_override: int = -1)` — [`L367`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L367) — Perform gather on indices within dynamic array start and end. — documented in [maxtext-layers-moe](../../../../../concepts/maxtext-layers-moe.md)

## Module values
- `_COMPILER_PARAMS` — [`L31`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L31)
- `_OUT_KW` — [`L29`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L29)
- `_SCRATCH_KW` — [`L30`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L30)

