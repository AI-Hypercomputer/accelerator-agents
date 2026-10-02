---
title: 'Module: jaxite/jaxite_ckks/bat_utils.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/bat_utils.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.bat_utils`/
symbols:
  matmul_bat_einsum: matmul_bat_einsum().
  basis_aligned_transformation: basis_aligned_transformation().
---
# Module: [`jaxite/jaxite_ckks/bat_utils.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/bat_utils.py)

## Functions
- `basis_aligned_transformation(matrix: jnp.ndarray, moduli: list[int])` — [`L38`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/bat_utils.py#L38) — Prepares a matrix for Basis Aligned Transformation (BAT). — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
- `matmul_bat_einsum(lhs: jax.Array, rhs: jax.Array, subscripts: str, merge_byte_dimension: bool = False)` — [`L10`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/bat_utils.py#L10) — Basis Aligned Transformation (BAT) based matrix multiplication.

