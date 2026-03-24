def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for a single row of a 2D depthwise convolution.

  This kernel computes one full row of the output feature map. It receives a
  horizontal slice of the input image and the complete convolution kernel.
  The core computation is performed using a manual loop over the output width,
  where each iteration computes a single output pixel via an element-wise
  product and sum.

  Args:
    x_ref: A reference to the input slice. The shape is
      (1, kernel_size_h, width, in_channels), corresponding to the part of the
      input image needed to compute one output row.
    kernel_ref: A reference to the convolution kernel. For this depthwise
      convolution, the shape is (kernel_size_h, kernel_size_w, 1, out_channels).
    out_ref: A reference to the output buffer. The kernel writes its result,
      a single output row of shape (1, 1, out_width, out_channels), here.
  """
  # Extract dimensions from the kernel and output reference shapes.
  kernel_size_h, kernel_size_w, _, _ = kernel_ref.shape
  out_width = out_ref.shape[2]
  in_channels = x_ref.shape[3]

  # Iterate over each horizontal position in the output row.
  for j in range(out_width):
    # Extract a patch from the input slice corresponding to the current
    # output position. The patch has the same dimensions as the kernel.
    # Stride is 1, so the patch starts at horizontal offset `j`.
    in_slice = jax.lax.dynamic_slice(x_ref, (0, 0, j, 0), (1, kernel_size_h, kernel_size_w, in_channels))

    # Perform the depthwise convolution for one output pixel.
    # This involves an element-wise product between the input patch and the
    # kernel, followed by a sum over the spatial dimensions (height, width).
    # Broadcasting rules handle the channel dimensions correctly.
    # in_slice: (1, kH, kW, C)
    # kernel_ref: (kH, kW, 1, C)
    # product: (1, kH, kW, C)
    # acc: (1, C) after summing over axes 1 and 2.
    acc = jnp.sum(in_slice * kernel_ref, axis=(1, 2))

    # Write the computed pixel (a vector of all channels) to the output buffer.
    # We squeeze the temporary batch dimension from `acc` to match the slice.
    out_ref[0, 0, j, :] = acc.squeeze(axis=0)
