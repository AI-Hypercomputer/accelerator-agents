def kernel(x_ref, w_ref, out_ref):
  """
  Pallas kernel for 2D convolution.

  This kernel computes a single tile of the output feature map. It takes a
  patch of the input `x_ref` and the full convolution weights `w_ref` to
  produce a tile of the output `out_ref`.

  Args:
    x_ref: A reference to the input patch. The shape is expected to be
      (1, bH + kernel_size - 1, bW + kernel_size - 1, in_channels).
    w_ref: A reference to the complete kernel weights. The shape is
      (kernel_size, kernel_size, in_channels, out_channels).
    out_ref: A reference to the output tile to be computed. The shape is
      (1, bH, bW, out_channels).
  """
  # Accumulator for the output tile, initialized to zeros.
  # Note: The accumulator shape should match the output tile's spatial and
  # channel dimensions. The batch dimension is handled at the end.
  acc = jnp.zeros((bH, bW, out_channels), dtype=jnp.float32)

  # Iterate over the kernel's spatial dimensions. These loops are unrolled
  # by the Pallas compiler.
  for kh in range(kernel_size):
    for kw in range(kernel_size):
      # Slice the input patch. The slice starts at (kh, kw) and has the
      # same spatial dimensions as the output tile (bH, bW).
      # x_slice will have shape (bH, bW, in_channels).
      x_slice = plax.slice(x_ref, (0, kh, kw, 0), (1, bH, bW, in_channels))
      x_slice = jnp.squeeze(x_slice, axis=0)

      # Get the corresponding weights for this kernel position.
      # w_slice will have shape (in_channels, out_channels).
      w_slice = w_ref[kh, kw, :, :]

      # Perform the dot product: (bH, bW, C_in) . (C_in, C_out)
      # The result has shape (bH, bW, C_out).
      update = jnp.dot(x_slice, w_slice, preferred_element_type=jnp.float32)

      # Accumulate the result.
      acc += update

  # Write the final accumulated result to the output reference, adding the
  # batch dimension back.
  out_ref[...] = jnp.expand_dims(acc, 0).astype(out_ref.dtype)
