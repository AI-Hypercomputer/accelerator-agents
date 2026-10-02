---
title: 'Module: pallas_forge/kernels/rmsnorm.py'
type: catalog
provenance: extracted
module: pallas_forge/kernels/rmsnorm.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.kernels.rmsnorm`/
symbols:
  fused_rmsnorm_residual: fused_rmsnorm_residual().
  TOKENS_PER_TILE: TOKENS_PER_TILE.
  rmsnorm_reference: rmsnorm_reference().
  _rmsnorm_residual_kernel: _rmsnorm_residual_kernel().
---
# Module: [`pallas_forge/kernels/rmsnorm.py`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/rmsnorm.py)

## Functions
- `_rmsnorm_residual_kernel(x_ref, residual_ref, weight_ref, out_ref, new_res_ref, *, eps: float)` — [`L34`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/rmsnorm.py#L34) — Fused RMSNorm + residual addition kernel (batched over 8 tokens).
- `fused_rmsnorm_residual(x: jax.Array, residual: jax.Array, weight: jax.Array, *, eps: float = 0.000001, block_size: int | None = None)` — [`L69`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/rmsnorm.py#L69) — Fused RMSNorm + residual addition.
- `rmsnorm_reference(x: jax.Array, weight: jax.Array, eps: float = 0.000001)` — [`L152`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/rmsnorm.py#L152) — Reference JAX implementation of RMSNorm (for testing).

## Module values
- `TOKENS_PER_TILE` — [`L31`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/rmsnorm.py#L31)

