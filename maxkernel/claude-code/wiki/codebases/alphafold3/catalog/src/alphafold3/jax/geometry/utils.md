---
title: 'Module: src/alphafold3/jax/geometry/utils.py'
type: catalog
provenance: extracted
module: src/alphafold3/jax/geometry/utils.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.jax.geometry.utils`/
symbols:
  unstack: unstack().
  weighted_mean: weighted_mean().
  angdiff: angdiff().
---
# Module: [`src/alphafold3/jax/geometry/utils.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/utils.py)

## Functions
- `angdiff(alpha: jnp.ndarray, beta: jnp.ndarray)` — [`L35`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/utils.py#L35) — Compute absolute difference between two angles.
- `unstack(value: jnp.ndarray, axis: int = -1)` — [`L28`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/utils.py#L28)
- `weighted_mean(*, weights: jnp.ndarray, value: jnp.ndarray, axis: int | Iterable[int] | None = None, eps: float = 1e-10)` — [`L42`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/utils.py#L42) — Computes weighted mean in a safe way that avoids NaNs. — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)

