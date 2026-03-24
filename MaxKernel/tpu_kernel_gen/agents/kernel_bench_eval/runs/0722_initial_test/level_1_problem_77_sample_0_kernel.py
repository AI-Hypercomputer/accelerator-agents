def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D transposed convolution.

  This kernel implements the transposed convolution by interpreting it as a
  standard convolution operating on a dilated input tensor with a flipped kernel.

  Args:
    x_ref: Reference to the input tensor.
    kernel_ref: Reference to the convolution kernel tensor.
    out_ref: Reference to the output tensor to be written to.
  """
  # Get the base indices for the output block this kernel is responsible for.
  # These correspond to the grid dimensions defined in the pallas_call.
  b = pl.program_id(0)
  od = pl.program_id(1)
  oh_base = pl.program_id(2) * bH
  ow_base = pl.program_id(3) * bW

  # Initialize an accumulator for the output block with zeros.
  # This accumulator holds the results for a tile of the output tensor.
  acc = jnp.zeros([bH, bW, out_channels], dtype=x_ref.dtype)

  # Iterate over each spatial element of the kernel.
  for kd in range(kernel_size):
    for kh in range(kernel_size):
      for kw in range(kernel_size):
        # Iterate over each pixel (y_h, y_w) within the output block's spatial tile.
        for y_h in range(bH):
          for y_w in range(bW):
            # Calculate the global output coordinates.
            oh = oh_base + y_h
            ow = ow_base + y_w

            # Calculate the corresponding coordinates in the dilated input space.
            # This is the core of the transposed convolution logic.
            dilated_d = od + padding - kd * dilation
            dilated_h = oh + padding - kh * dilation
            dilated_w = ow + padding - kw * dilation

            # Calculate the original input coordinates.
            id = dilated_d // stride
            ih = dilated_h // stride
            iw = dilated_w // stride

            # Create a mask to check for validity. An input is valid if it's not
            # from stride-based dilation and is within the original input bounds.
            is_on_grid = (dilated_d % stride == 0) & (dilated_h % stride == 0) & (dilated_w % stride == 0)
            is_in_bounds = (0 <= id < depth) & (0 <= ih < height) & (0 <= iw < width)
            mask = is_on_grid & is_in_bounds

            # The body of the conditional update.
            def update_fn(acc):
              # The kernel is flipped spatially.
              flipped_kd = kernel_size - 1 - kd
              flipped_kh = kernel_size - 1 - kh
              flipped_kw = kernel_size - 1 - kw

              # Load the input vector and the corresponding flipped kernel matrix.
              x_vec = x_ref[b, id, ih, iw, :]
              kernel_mat = kernel_ref[flipped_kd, flipped_kh, flipped_kw, :, :]

              # Perform the dot product and return the result. On TPU, pl.dot
              # expects 2D inputs for the MXU. We expand the input vector
              # to a row vector and squeeze the output.
              update = pl.dot(x_vec[None, :], kernel_mat)[0]
              return acc.at[y_h, y_w, :].add(update)

            # Use lax.cond to conditionally execute the update. This avoids
            # Python 'if' statements on tracers and prevents out-of-bounds
            # memory access by only calling update_fn when the mask is True.
            acc = jax.lax.cond(mask, update_fn, lambda acc: acc, acc)

  # Write the computed accumulator block to the final output tensor.
  out_ref[b, od, oh_base : oh_base + bH, ow_base : ow_base + bW, :] = acc
