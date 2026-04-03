def kernel(x_ref, kernel_ref, y_ref):
  # Hardcoded parameters from the problem description for clarity
  kernel_size = 3
  in_channels = 32
  out_channels = 64
  height_in = 64
  width_in = 128
  stride = 5
  padding = 1
  dilation = 2
  # This is now a loop bound, not a grid index
  width_out = 638

  # Each program instance computes one output ROW (h_out)
  h_out = pl.program_id(1)
  # b_idx is implicit in the data slices we receive
  # b_idx = pl.program_id(0)

  # We need to loop over the width of the output row
  def w_out_loop_body(w_out, _):
    # Accumulator for the output pixel over all output channels
    acc = jnp.zeros((out_channels,), dtype=y_ref.dtype)

    # Loop over the kernel's spatial dimensions (height)
    def kh_loop_body(k_h, acc_kh):
      # Loop over the kernel's spatial dimensions (width)
      def kw_loop_body(k_w, acc_kw):
        # Determine the corresponding input pixel coordinates
        h_in_nom = h_out + padding - k_h * dilation
        w_in_nom = w_out + padding - k_w * dilation

        # Check if this kernel position contributes to the output pixel.
        # 1. The input coordinates must align with the stride.
        is_valid_stride = (h_in_nom % stride == 0) & (w_in_nom % stride == 0)
        h_in = h_in_nom // stride
        w_in = w_in_nom // stride

        # 2. The input coordinates must be within the input tensor bounds.
        is_valid_bounds = (h_in >= 0) & (h_in < height_in) & (w_in >= 0) & (w_in < width_in)

        is_valid_pixel = is_valid_stride & is_valid_bounds

        def true_fn():
          # If valid, fetch the input vector and kernel slice
          # x_ref is 3D: (H_in, W_in, C_in) due to squeezing
          x_slice = x_ref[h_in, w_in, :]
          kernel_slice = kernel_ref[k_h, k_w, :, :]
          # Compute the contribution and return it
          return jnp.dot(x_slice, kernel_slice)

        def false_fn():
          # If not valid, the contribution is zero
          return jnp.zeros_like(acc_kw)

        # Add the contribution for this kernel position to the accumulator
        return acc_kw + jax.lax.cond(is_valid_pixel, true_fn, false_fn)

      # Run the inner loop over kernel width
      return jax.lax.fori_loop(0, kernel_size, kw_loop_body, acc_kh)

    # Run the outer loop over kernel height
    acc = jax.lax.fori_loop(0, kernel_size, kh_loop_body, acc)

    # Write the final accumulated values to the output reference.
    # y_ref has shape (width_out, out_channels) due to squeezing.
    # We write the computed pixel `acc` of shape (out_channels,)
    # at the index (w_out, :).
    pl.store(y_ref, (w_out, slice(None)), acc)
    return None  # No carry needed for the w_out loop

  jax.lax.fori_loop(0, width_out, w_out_loop_body, None)
