def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D transposed convolution.

  This kernel computes a single row of output values (a tensor of size
  (width_out, out_channels)) for the transposed convolution. The grid for the
  pallas_call iterates over the batch and the first two spatial output
  dimensions (depth, height).

  Args:
    x_ref: A reference to the entire padded input tensor.
    kernel_ref: The entire kernel tensor.
    out_ref: A reference to a single output row to be written to.
  """
  b = pl.program_id(0)
  d = pl.program_id(1)
  h = pl.program_id(2)

  k_d, k_h, k_w = kernel_size

  # This kernel computes a whole row along the 'w' dimension to satisfy
  # TPU block size constraints.
  for w in range(width_out):
    acc = jnp.zeros(out_channels, dtype=x_ref.dtype)
    # The core logic implements a 3D convolution on a padded input, which is
    # equivalent to a transposed convolution.
    # output[d,h,w] = sum_{kd,kh,kw} x_padded[d+kd, h+kh, w+kw] * kernel_flipped[kd,kh,kw]
    # This is implemented by flipping the access into the kernel, which is equivalent.
    for kd in range(k_d):
      for kh in range(k_h):
        for kw in range(k_w):
          # Access the input vector for this position
          x_vec = x_ref[b, d + kd, h + kh, w + kw, :]
          # Access the corresponding flipped kernel slice
          k_mat = kernel_ref[k_d - 1 - kd, k_h - 1 - kh, k_w - 1 - kw, :, :]
          acc += jnp.dot(jnp.expand_dims(x_vec, 0), k_mat).squeeze(axis=0)
    # Write the accumulated value to the output row.
    out_ref[0, 0, 0, w, :] = acc
