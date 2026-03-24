def kernel(x_ref, y_ref, out_ref):
  """Pallas kernel for tril(matmul(A, B))."""
  i, j = pl.program_id(0), pl.program_id(1)

  # The computation is C = tril(A @ B). We can analyze this block-wise.
  # Let C_ij be the block of C at index (i, j).
  #
  # We use pl.cond to handle control flow, as standard Python `if` statements
  # cannot be used with JAX tracers like `i` and `j`.

  def upper_triangle_case(_):
    # Case 1: j > i
    # The block C_ij is in the strict upper triangle of the matrix C.
    # Applying jnp.tril to the final matrix will make this entire block zero.
    # We can optimize by not computing the matmul at all for these blocks.
    return jnp.zeros_like(out_ref)

  def lower_or_diagonal_case(_):
    # For blocks on or below the diagonal (j <= i), we must compute the matmul.
    result = x_ref[...] @ y_ref[...]

    def diagonal_case(_):
      # Case 2: i == j
      # The block C_ij is on the main diagonal of the matrix C.
      # We need to apply jnp.tril to the result of the block matmul.
      return jnp.tril(result)

    def lower_triangle_case(_):
      # Case 3: j < i
      # The block C_ij is in the strict lower triangle of the matrix C.
      # Applying jnp.tril to the final matrix will preserve this entire block.
      return result

    # Nested pl.cond to differentiate between diagonal and lower triangle cases.
    return pl.cond(i == j, diagonal_case, lower_triangle_case, operand=None)

  # Top-level pl.cond to handle the main optimization.
  out_ref[...] = pl.cond(j > i, upper_triangle_case, lower_or_diagonal_case, operand=None)
