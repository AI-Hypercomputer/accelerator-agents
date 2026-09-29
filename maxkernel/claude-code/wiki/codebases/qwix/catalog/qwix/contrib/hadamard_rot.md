---
title: 'Module: qwix/contrib/hadamard_rot.py'
type: catalog
provenance: extracted
module: qwix/contrib/hadamard_rot.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.hadamard_rot`/
symbols:
  hadamard_rotate_inputs: hadamard_rotate_inputs().
  _create_hadamard_matrix: _create_hadamard_matrix().
  _apply_hadamard_lhs_rhs: _apply_hadamard_lhs_rhs().
  _create_base_hadamard_matrix: _create_base_hadamard_matrix().
  _apply_hadamard_lhs: _apply_hadamard_lhs().
  _apply_hadamard_rhs: _apply_hadamard_rhs().
---
# Module: [`qwix/contrib/hadamard_rot.py`](../../../../../../raw/code/qwix/qwix/contrib/hadamard_rot.py)

## Functions
- `_apply_hadamard_lhs(x: jax.Array, had: jax.Array, reduction_dim: int)` — [`L99`](../../../../../../raw/code/qwix/qwix/contrib/hadamard_rot.py#L99) — Applies the lhs Hadamard transform to x along the reduction dimension.
- `_apply_hadamard_lhs_rhs(act: jax.Array, weight: jax.Array, had: jax.Array, lhs_reduction_dim: int, rhs_reduction_dim: int)` — [`L120`](../../../../../../raw/code/qwix/qwix/contrib/hadamard_rot.py#L120) — Applies the Hadamard matrix to the LHS and RHS of the weight matrix.
- `_apply_hadamard_rhs(x: jax.Array, had: jax.Array, reduction_dim: int)` — [`L109`](../../../../../../raw/code/qwix/qwix/contrib/hadamard_rot.py#L109) — Applies the rhs Hadamard transform to x along the reduction dimension.
- `_create_base_hadamard_matrix(power: int)` — [`L35`](../../../../../../raw/code/qwix/qwix/contrib/hadamard_rot.py#L35) — Returns a bfloat16 Hadamard matrix of size 2^power x 2^power.
- `_create_hadamard_matrix(power: int, key: jax.Array | None, *, row_sign_flip: bool, col_sign_flip: bool, dtype: jnp.dtype = jnp.bfloat16)` — [`L45`](../../../../../../raw/code/qwix/qwix/contrib/hadamard_rot.py#L45) — Returns a Hadamard matrix of size 2^power x 2^power with random sign flips.
- `hadamard_rotate_inputs(act: jax.Array, weight: jax.Array, key: jax.Array | None, *, row_sign_flip: bool, col_sign_flip: bool, lhs_reduction_dim: int, rhs_reduction_dim: int)` — [`L133`](../../../../../../raw/code/qwix/qwix/contrib/hadamard_rot.py#L133) — Constructs a Hadamard matrix and applies it to the inputs of the weight matrix.

