def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for grouped transposed convolution."""
  # Hardcoded parameters from the nn.ConvTranspose layer configuration.
  strides = (2, 3)
  padding = ((1, 1), (2, 2))
  dilation = (2, 1)

  # The `lhs` and `rhs` are slices of the full input and kernel tensors,
  # corresponding to a single group. We therefore perform a standard
  # (non-grouped) convolution on these slices.
  result = jax.lax.conv_transpose(lhs=x_ref, rhs=kernel_ref, strides=strides, padding=padding, rhs_dilation=dilation)
  out_ref[...] = result
