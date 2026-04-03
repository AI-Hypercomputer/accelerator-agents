def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D transposed convolution.

  This kernel computes a block of the output tensor. It iterates over the
  spatial dimensions of the output block. For each output position (d, h, w),
  it slices the corresponding input window from `x_ref` and computes the
  dot product with the convolution kernel.

  Args:
    x_ref: A reference to a block of the input tensor.
    kernel_ref: A reference to the entire convolution kernel.
    out_ref: A reference to the output block to be computed.
  """
  # Get the spatial dimensions of the output block and the kernel.
  # Note: b_b (batch block size) is 1 in this configuration.
  _, b_d, b_h, b_w, _ = out_ref.shape
  kd, kh, kw, _, _ = kernel_ref.shape

  # Iterate over each point in the output block's spatial dimensions.
  for d in range(b_d):
    for h in range(b_h):
      for w in range(b_w):
        # For each output pixel (d, h, w), we take a slice of the input
        # block of size (kd, kh, kw) starting at (d, h, w).
        # Since b_b is 1, we can index the batch dimension with 0.
        input_slice = x_ref[0, d : d + kd, h : h + kh, w : w + kw, :]

        # Compute the dot product between the input slice and the kernel.
        # This is the core convolution operation for a single output pixel.
        # The contracting dimensions are the spatial (0,1,2) and
        # input channel (3) dimensions.
        # lhs shape: (kd, kh, kw, in_channels)
        # rhs shape: (kd, kh, kw, in_channels, out_channels)
        # result shape: (out_channels,)
        out_channels_val = jax.lax.dot_general(
          lhs=input_slice,
          rhs=kernel_ref[...],
          dimension_numbers=(((0, 1, 2, 3), (0, 1, 2, 3)), ((), ())),
        )
        # Assign the computed vector to the correct output pixel.
        out_ref[0, d, h, w, :] = out_channels_val
