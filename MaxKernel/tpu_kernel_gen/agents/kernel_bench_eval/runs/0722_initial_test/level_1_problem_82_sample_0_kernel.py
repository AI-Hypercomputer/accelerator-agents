def kernel(x_ref, kernel_weights_ref, out_ref):
  """Pallas kernel for depthwise convolution.

  This kernel computes a single output value for a depthwise convolution.
  The grid is set up so that each kernel instance corresponds to one output
  pixel in one channel of the output tensor.

  Args:
    x_ref: A reference to a 4D slice (patch) of the input tensor `x` of
      shape (1, kernel_size, kernel_size, 1).
    kernel_weights_ref: A reference to a 4D slice of the kernel weights
      tensor of shape (kernel_size, kernel_size, 1, 1) for the current channel.
    out_ref: A reference to a scalar element in the output tensor where the
      result will be written.
  """
  # The core operation of convolution is a dot product between the input patch
  # and the kernel filter. This is achieved by an element-wise multiplication
  # followed by a sum of the resulting elements.
  # Index the singleton dimensions to perform a 2D convolution.
  x_patch = x_ref[0, :, :, 0]
  w_patch = kernel_weights_ref[:, :, 0, 0]
  conv_result = jnp.sum(x_patch * w_patch)

  # Write the computed scalar value to the designated output location.
  out_ref[...] = conv_result
