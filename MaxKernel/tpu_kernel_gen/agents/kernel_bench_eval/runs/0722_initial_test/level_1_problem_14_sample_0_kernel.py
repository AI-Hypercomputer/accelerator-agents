def kernel(a_ref, b_ref, c_ref):
  """Pallas kernel to compute C = jnp.triu(jnp.matmul(A, B))."""
  i, j = pl.program_id(0), pl.program_id(1)
  block_size = a_ref.shape[0]

  # A block of the output matrix C is in the lower triangle if its
  # block row index `i` is greater than its block column index `j`.
  # In this case, the entire block should be zero due to the jnp.triu.
  if i > j:
    c_ref[...] = 0
    return

  # Otherwise, we are in the upper triangle (or the diagonal)
  # and need to compute the matrix multiplication for this block.
  result = a_ref[...] @ b_ref[...]

  # If the block is on the main diagonal (i == j), we must apply
  # jnp.triu to the resulting block itself to zero out its lower part.
  # Otherwise (i < j), the entire block is part of the final result.
  if i == j:
    # Manually implement triu for the diagonal block
    rows = jnp.arange(block_size)
    cols = jnp.arange(block_size)
    mask = rows[:, None] <= cols[None, :]
    final_result = jnp.where(mask, result, 0)
  else:
    final_result = result
  c_ref[...] = final_result
