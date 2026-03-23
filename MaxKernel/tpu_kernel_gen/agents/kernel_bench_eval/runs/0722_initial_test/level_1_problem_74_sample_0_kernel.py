def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for 1D transposed convolution.

  This kernel computes a block of the output tensor at a time. It iterates
  through the convolution kernel's positions and, for each, calculates the
  corresponding input data required. It uses vectorized operations to gather
  input data and perform the matrix multiplication with the kernel slice.
  Masking is applied to handle boundary conditions correctly, ensuring no
  out-of-bounds memory access occurs for either the input or the output.

  Args:
    x_ref: Reference to the input tensor block.
    kernel_ref: Reference to the convolution kernel.
    out_ref: Reference to the output tensor block to be written to.
  """
  # Get the index for the current output block along the length dimension.
  j = pl.program_id(1)

  # Define convolution parameters (these are fixed for this specific kernel).
  kernel_size = 5
  length = 256
  stride = 1
  padding = 0
  dilation = 3

  # Determine block size and output channels from the output reference shape.
  bL = out_ref.shape[1]
  out_channels = out_ref.shape[2]

  # Calculate the total length of the final output tensor.
  output_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

  # Create an array of global indices for the current output block.
  l_out_global = j * bL + pl.arange(bL)
  # Create a mask to handle the last block, which might be smaller than bL.
  out_mask = l_out_global < output_length

  # The main loop iterates over the kernel's spatial dimension.
  def body_loop(l_kernel, acc):
    # Calculate the corresponding input indices for the current output indices.
    l_in = l_out_global - l_kernel * dilation

    # Create a mask to ensure we only read from valid input memory locations.
    in_mask = (l_in >= 0) & (l_in < length)

    # Clamp indices to prevent out-of-bounds errors during the gather.
    l_in_clamped = jnp.clip(l_in, 0, length - 1)

    # Gather the relevant input data. x_ref[0] is used as the batch dimension
    # is handled by the grid and is 1 within the kernel.
    x_gathered = x_ref[0, l_in_clamped, :]

    # Combine input and output masks to determine valid contributions.
    # A contribution is only valid if both its source and destination are in bounds.
    combined_mask = (in_mask & out_mask)[:, jnp.newaxis]

    # Apply the mask to zero out contributions from invalid positions.
    x_masked = x_gathered * combined_mask

    # Perform the matrix multiplication between the masked inputs and the kernel slice.
    contribution = x_masked @ kernel_ref[l_kernel]

    # Accumulate the results.
    return acc + contribution

  # Initialize an accumulator for the output block with zeros.
  # The shape is (bL, out_channels).
  out_block_acc = jnp.zeros_like(out_ref[0, ...])

  # Run the accumulation loop over the kernel dimension.
  out_block = lax.fori_loop(0, kernel_size, body_loop, out_block_acc)

  # Write the final computed block to the output reference, adding the batch dimension back.
  out_ref[...] = out_block[jnp.newaxis, :, :]


# Computation
# Calculate output shape
output_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
output_shape = (batch_size, output_length, out_channels)

# Define block size for the length dimension
bL = 128

# Pallas kernel invocation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, (output_length + bL - 1) // bL),
  in_specs=[
    pl.BlockSpec(block_shape=(1, length, in_channels), index_map=lambda i, j: (i, 0, 0)),
    pl.BlockSpec(block_shape=kernel_param.shape, index_map=lambda i, j: (0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, bL, out_channels), index_map=lambda i, j: (i, j * bL, 0)),
)(x, kernel_param).block_until_ready()
