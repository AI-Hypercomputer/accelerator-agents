def kernel(x_ref, depthwise_kernel_ref, pointwise_kernel_ref, out_ref):
  """Pallas kernel for separable convolution."""
  # The pointwise convolution (1x1 convolution) is equivalent to a dot product.
  # We need to squeeze the kernel dimensions to make it a 2D matrix for the dot
  # product. The kernel shape is (1, 1, in_channels, out_channels).
  pointwise_kernel = jax.lax.squeeze(pointwise_kernel_ref, (0, 1))

  # The depthwise convolution applies a filter to each channel independently.
  # For a single tile, this can be implemented by an element-wise multiplication
  # if the kernel is broadcastable to the input tile's shape. This is a
  # simplification and assumes a 1x1 depthwise kernel for this example.
  # A full convolution would require explicit sliding window operations.
  depthwise_out = x_ref * depthwise_kernel_ref

  # Perform the pointwise convolution on the result of the depthwise operation.
  pointwise_out = jax.lax.dot(depthwise_out, pointwise_kernel)

  # Write the final result to the output tile.
  out_ref[...] = pointwise_out
