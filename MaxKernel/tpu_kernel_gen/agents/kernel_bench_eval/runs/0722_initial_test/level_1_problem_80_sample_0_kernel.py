def kernel(x_ref, kernel_ref, out_ref):
  # Unpack constants from the kernel's closure.
  # These values are defined in the scope where `pallas_call` is invoked.
  # This is a common and convenient pattern for passing static parameters.
  (kernel_height, kernel_width) = kernel_size
  (dilation_h, dilation_w) = dilation
  (pad_h_start, _) = padding[0]
  (pad_w_start, _) = padding[1]

  # Each program instance computes one output pixel across all output channels.
  # The grid is (batch_size, out_height, out_width).
  b = pl.program_id(0)
  oh = pl.program_id(1)
  ow = pl.program_id(2)

  # Initialize an accumulator register for the output pixel.
  # It will hold the computed values for all output channels.
  # The shape is (out_channels,).
  acc = jnp.zeros((out_channels,), dtype=x_ref.dtype)

  # Iterate over the spatial dimensions of the convolution kernel.
  for kh in range(kernel_height):
    for kw in range(kernel_width):
      # Calculate the corresponding input coordinates for the current kernel position.
      # The formula accounts for stride, dilation, and padding.
      # input_coord = output_coord * stride + kernel_coord * dilation - padding_start
      ih = oh * stride + kh * dilation_h - pad_h_start
      iw = ow * stride + kw * dilation_w - pad_w_start

      # Perform the multiply-accumulate operation only if the calculated
      # input coordinates are within the bounds of the original input tensor.
      # This check effectively handles the padding, as any access to the
      # padded region is skipped (contributing a value of zero).
      in_bounds = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
      # Load the slice of the input tensor corresponding to the calculated coordinates.
      # This slice contains all input channels for a single spatial point.
      # Shape: (in_channels,)
      input_slice = pl.load(x_ref, (b, ih, iw, slice(None)), mask=in_bounds, other=0.0)

      # Load the corresponding slice of the kernel weights.
      # Shape: (in_channels, out_channels)
      kernel_slice = pl.load(kernel_ref, (kh, kw, slice(None), slice(None)))

      # Perform a dot product between the input slice and the kernel slice.
      # This calculates the contribution of this input patch to all output channels.
      # jnp.dot((in_channels,), (in_channels, out_channels)) -> (out_channels,)
      acc += jnp.dot(input_slice, kernel_slice)

  # Write the final accumulated values to the output buffer.
  # The output reference `out_ref` has a shape that will broadcast the
  # accumulator correctly.
  out_ref[...] = acc
