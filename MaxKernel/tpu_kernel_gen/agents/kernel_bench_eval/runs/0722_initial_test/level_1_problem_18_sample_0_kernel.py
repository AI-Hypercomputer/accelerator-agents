def kernel(x_ref, y_ref, out_ref):
  """Pallas kernel to compute C = A.T @ B.T.

  Args:
    x_ref: A reference to a block of input matrix A.
    y_ref: A reference to a block of input matrix B.
    out_ref: A reference to a block of the output matrix C.
  """
  # x_ref is a block of A.T with shape (bM, K).
  # y_ref is a block of B.T with shape (K, bN).
  # The matrix multiplication gives a result of shape (bM, bN), which
  # matches the shape of the output block.
  # The data is loaded in the appropriate layout for the matmul, so no
  # explicit transpose is needed.
  x = x_ref[...]
  y = y_ref[...]
  out_ref[...] = jnp.matmul(x, y)
