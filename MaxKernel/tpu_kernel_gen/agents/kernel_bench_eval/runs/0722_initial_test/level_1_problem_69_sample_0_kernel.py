def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D transposed convolution.

  This kernel implements a "scatter" approach where each program, corresponding
  to a single input pixel, calculates its contribution to the output and adds
  it to the appropriate output tile.

  Args:
    x_ref: A reference to a slice of the input tensor.
      Shape: (1, 1, 1, in_channels)
    kernel_ref: A reference to the full convolution kernel weights.
      Shape: (kernel_h, kernel_w, in_channels, out_channels)
    out_ref: A reference to a tile of the output tensor to which results are
      accumulated.
      Shape: (1, kernel_h, kernel_w, out_channels)
  """
  # Load the slice of x into a local variable.
  x_val = x_ref[...]
  # Squeeze the leading dimensions of x_val to get a vector of input channels.
  # The resulting shape is (in_channels,).
  x_vec = jnp.squeeze(x_val, axis=(0, 1, 2))

  # Perform the core transposed convolution operation for a single input pixel.
  # This is equivalent to multiplying the input channel vector (x_vec) by the
  # kernel and accumulating the results. A tensor dot product achieves this
  # efficiently by contracting the `in_channels` dimension from both tensors.
  # The contraction axes are axis 0 for x_vec and axis 2 for kernel_ref.
  # The resulting shape of the update is (kernel_h, kernel_w, out_channels).
  update = jnp.tensordot(x_vec, kernel_ref, axes=([0], [2]))

  # The output of the tensordot operation needs to be reshaped to match the
  # shape of the output reference block, which includes a singleton batch dim.
  update = update.reshape(out_ref.shape)

  # Atomically add the computed update to the output tile. Pallas handles the
  # accumulation correctly even when different kernel instances write to
  # overlapping regions of the output, as is common with strides < kernel_size.
  out_ref[...] += update
