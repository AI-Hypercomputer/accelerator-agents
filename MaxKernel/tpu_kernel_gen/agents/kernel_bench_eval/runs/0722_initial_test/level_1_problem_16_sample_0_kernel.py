def kernel(a_ref, b_ref, c_ref):
  """Pallas kernel for the computation C = A.T @ B.

  This kernel calculates a tile of the output matrix C. The grid of programs
  iterates over the M and N dimensions of C. For each program, the
  corresponding slices of A and B are loaded.

  Args:
    a_ref: A reference to a block of matrix A with shape (K, bM).
    b_ref: A reference to a block of matrix B with shape (K, bN).
    c_ref: A reference to a block of the output matrix C with shape (bM, bN),
      which this kernel will populate.
  """
  acc = jnp.zeros((bM, bN), dtype=c_ref.dtype)
  bK = 128
  for k in range(K // bK):
    a_block = pl.load(a_ref, (k * bK, 0), block_shape=(bK, bM))
    b_block = pl.load(b_ref, (k * bK, 0), block_shape=(bK, bN))
    acc = pl.dot(jnp.swapaxes(a_block, 0, 1), b_block, acc)
  c_ref[...] = acc
