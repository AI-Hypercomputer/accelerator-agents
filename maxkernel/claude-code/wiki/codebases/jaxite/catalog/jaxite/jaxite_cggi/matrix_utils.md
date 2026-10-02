---
title: 'Module: jaxite/jaxite_cggi/matrix_utils.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/matrix_utils.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.matrix_utils`/
symbols:
  toeplitz_poly_mul: toeplitz_poly_mul().
  monomial_mul: monomial_mul().
  toeplitz: toeplitz().
  i32_as_u8_matmul: i32_as_u8_matmul().
  smul_as_dense_gemv_bat: smul_as_dense_gemv_bat().
  poly_mul: poly_mul().
  monomial_mul_list: monomial_mul_list.
  poly_dot_product: poly_dot_product().
  x_power_n_minus_1: x_power_n_minus_1().
  hpmatmul_offline_compile_bat: hpmatmul_offline_compile_bat().
  toeplitz_kernelized: toeplitz_kernelized().
  int32_to_int8_arr: int32_to_int8_arr().
  scale_by_x_power_n_minus_1: scale_by_x_power_n_minus_1().
  hpmatmul_conv_adapt_outer_product: hpmatmul_conv_adapt_outer_product().
  poly_mul_list: poly_mul_list.
  poly_mul_const_list: poly_mul_const_list.
  poly_mul_const_matrix: poly_mul_const_matrix.
  integer_div: integer_div().
  chunk_decomposition: chunk_decomposition().
  hpmatmul_golden: hpmatmul_golden().
  _generate_sign_matrix: _generate_sign_matrix().
  smul_as_dense_gemv_bat_jax: smul_as_dense_gemv_bat_jax().
  monomial_mul_matrix: monomial_mul_matrix.
  rechunkify_after_chunkwise_add: rechunkify_after_chunkwise_add().
  hpmatmul_bat_adapt: hpmatmul_bat_adapt().
  toeplitz_kernelized._toeplitz: toeplitz_kernelized()._toeplitz().
  hpmatmul_conv_adapt_conv: hpmatmul_conv_adapt_conv().
---
# Module: [`jaxite/jaxite_cggi/matrix_utils.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py)

## Functions
- `_generate_sign_matrix(n: int)` — [`L649`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L649) — Generates a sign matrix with 1s below the diagonal and -1 above.
- `_toeplitz(inp_ref, out_ref)` — [`L712`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L712)
- `chunk_decomposition(x, chunkwidth=8)` — [`L204`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L204) — Precision-level data conversion.
- `hpmatmul_bat_adapt(lhs: jax.Array, y: jax.Array)` — [`L616`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L616) — Input (m, n) Left Matrix -> (m, n, p, q) Left Matrix, where each element in the original (m, n) matrix is replaced by a (p, q) matrix.
- `hpmatmul_conv_adapt_conv(x: jnp.ndarray, y: jnp.ndarray)` — [`L138`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L138) — Interleaved u8 matmul with padded 1D convolution.
- `hpmatmul_conv_adapt_outer_product(x: jax.Array, y: jax.Array)` — [`L103`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L103) — Interleaved u8 matmul with fused einsum kernels.
- `hpmatmul_golden(mat_a, mat_b, modulus_32)` — [`L630`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L630)
- `hpmatmul_offline_compile_bat(mat_a, q)` — [`L572`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L572) — Convert the input (m,n) matrix into (m,n,p,q), i.e.
- `i32_as_u8_matmul(lhs: jnp.ndarray, rhs: jnp.ndarray)` — [`L74`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L74) — Multiply an (n,) by an (n, k) i32 matrix using only i8 ops.
- `int32_to_int8_arr(arr: jnp.ndarray)` — [`L68`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L68) — Decompose an int32 matrix into u8s.
- `integer_div(values: jnp.ndarray, divisor: jnp.uint32)` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L15) — Performs integer division with rounding for positive integers X, d.
- `monomial_mul(poly: jnp.ndarray, degree: jnp.uint32, log_modulus: jnp.uint32)` — [`L763`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L763) — Computes `poly * X^degree mod (X^N + 1)` where N = len(poly).
- `poly_dot_product(poly_vec1: jnp.ndarray, poly_vec2: jnp.ndarray)` — [`L840`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L840) — Compute a dot product of two vectors of polynomials.
- `poly_mul(a: jnp.ndarray, b: jnp.ndarray)` — [`L756`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L756) — Computes a poly multiplication mod (X^N + 1) where N = len(a). — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
- `rechunkify_after_chunkwise_add(arr_a, chunkwidth)` — [`L233`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L233) — Rechunkify after chunkwise add.
- `scale_by_x_power_n_minus_1(power: jnp.int32, matrix: jnp.ndarray, log_modulus: int)` — [`L848`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L848) — An optimized poly mul for scaling a matrix of polynomials by x^n - 1.
- `smul_as_dense_gemv_bat(x, total_in_precision=32, chunkwidth=8, q=4294967291)` — [`L349`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L349) — Implements Basis Align Transformation (BAT) for dense matrices.
- `smul_as_dense_gemv_bat_jax(x, q=4294967291)` — [`L466`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L466) — This is the implementation of bat; Major improvement to achieve dense matrix.
- `toeplitz(x: jnp.ndarray)` — [`L658`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L658) — Generates a cyclic matrix with each row of the input shifted.
- `toeplitz_kernelized(x: jnp.ndarray)` — [`L686`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L686) — Use pltpu.roll op to implement toeplitz + sign matrix.
- `toeplitz_poly_mul(a: jnp.ndarray, b: jnp.ndarray)` — [`L740`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L740) — Computes a poly multiplication mod (X^N + 1) where N = len(a).
- `x_power_n_minus_1(n: jnp.uint32, poly_mod_deg: jnp.uint32)` — [`L45`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L45) — Construct a polynomial of the form x^d - 1 for an input power d.

## Module values
- `monomial_mul_list` — [`L818`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L818)
- `monomial_mul_matrix` — [`L835`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L835)
- `poly_mul_const_list` — [`L826`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L826)
- `poly_mul_const_matrix` — [`L830`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L830)
- `poly_mul_list` — [`L822`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L822)

