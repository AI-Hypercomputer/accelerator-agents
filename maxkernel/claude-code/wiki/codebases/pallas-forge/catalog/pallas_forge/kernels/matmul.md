---
title: 'Module: pallas_forge/kernels/matmul.py'
type: catalog
provenance: extracted
module: pallas_forge/kernels/matmul.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.kernels.matmul`/
symbols:
  tiled_matmul: tiled_matmul().
  _matmul_kernel: _matmul_kernel().
  _matmul_kernel._reset: _matmul_kernel()._reset().
---
# Module: [`pallas_forge/kernels/matmul.py`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/matmul.py)

## Functions
- `_matmul_kernel(x_ref, w_ref, o_ref)` — [`L32`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/matmul.py#L32) — Pallas kernel for one (i, j, k) cell of the 3D matmul grid.
- `_reset()` — [`L45`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/matmul.py#L45)
- `tiled_matmul(x: jax.Array, w: jax.Array, *, block_m: int = 128, block_k: int = 128, block_n: int = 128, num_stages: int = 2)` — [`L52`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/matmul.py#L52) — Tiled matrix multiplication using Pallas.

