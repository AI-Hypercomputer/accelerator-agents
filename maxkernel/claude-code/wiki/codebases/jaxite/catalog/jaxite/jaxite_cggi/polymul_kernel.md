---
title: 'Module: jaxite/jaxite_cggi/polymul_kernel.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/polymul_kernel.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.polymul_kernel`/
symbols:
  negacyclic_vector_matrix_polymul_bat: negacyclic_vector_matrix_polymul_bat().
  negacyclic_vector_matrix_polymul: negacyclic_vector_matrix_polymul().
  fallback_vector_matrix_polymul: fallback_vector_matrix_polymul.
  fallback_i32_matmul: fallback_i32_matmul.
  _decomposed_vector_matrix_polymul: _decomposed_vector_matrix_polymul().
  _decomposed_vector_matrix_polymul.vec_mat_polymul_kernel: _decomposed_vector_matrix_polymul().vec_mat_polymul_kernel().
  i32_matmul_unreduced: i32_matmul_unreduced().
  i32_matmul: i32_matmul().
  _i32_matmul_unreduced: _i32_matmul_unreduced().
  bat_matmul: bat_matmul().
  _i32_matmul_unreduced_cggi: _i32_matmul_unreduced_cggi().
  negacyclic_vector_matrix_polymul_bat._toeplitz_chunk: negacyclic_vector_matrix_polymul_bat()._toeplitz_chunk().
---
# Module: [`jaxite/jaxite_cggi/polymul_kernel.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py)

## Functions
- `_decomposed_vector_matrix_polymul(poly_vec1: jnp.ndarray, poly_mat2: jnp.ndarray)` — [`L140`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L140) — Computes decomposed vector-matrix polynomial multiplication.
- `_i32_matmul_unreduced(lhs, rhs)` — [`L32`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L32) — Performs an unreduced 32-bit integer matrix multiplication.
- `_i32_matmul_unreduced_cggi(lhs, rhs)` — [`L85`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L85) — Performs a 32-bit integer matrix multiplication with CGGI optimization.
- `_toeplitz_chunk(poly_vec1, vec_toeplitz)` — [`L290`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L290)
- `bat_matmul(lhs: jax.Array, y: jax.Array)` — [`L63`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L63) — Perform matrix multiplication between u8(c, m, n, 4, 4)@u32(c, n, k).
- `i32_matmul(lhs, rhs)` — [`L342`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L342) — A helper to isolate the matmul part of the kernel to test in isolation.
- `i32_matmul_unreduced(lhs, rhs, out)` — [`L337`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L337) — A helper to isolate the matmul part of the kernel to test in isolation.
- `negacyclic_vector_matrix_polymul(vec: jnp.ndarray, matrix: jnp.ndarray, decomposition_log_base: int)` — [`L219`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L219) — Computes a vector-matrix poly multiplication mod (X^N + 1).
- `negacyclic_vector_matrix_polymul_bat(poly_vec1: jnp.ndarray, poly_mat2: jnp.ndarray)` — [`L252`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L252) — Perform matrix multiplication between u8(c, m, n, 4, 4)@u32(c, n, k).
- `vec_mat_polymul_kernel(vec_ref, mat_ref, out_ref)` — [`L171`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L171) — Pallas kernel for polynomial multiplication.

## Module values
- `fallback_i32_matmul` — [`L27`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L27)
- `fallback_vector_matrix_polymul` — [`L17`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L17)

