def kernel(x_ref, out_ref):
  """Pallas kernel for 1D max pooling."""

  def loop_body(j, _):
    # This inner function processes one output element per iteration of the outer loop.
    max_vals = jnp.full((features,), -jnp.inf, dtype=x_ref.dtype)

    def window_loop(k, current_max):
      idx = j * stride - padding + k * window_dilation
      # Use lax.cond to safely handle out-of-bounds reads.
      return lax.cond(
        (idx >= 0) & (idx < sequence_length),
        # If in bounds, read from x_ref and update the max.
        lambda: jnp.maximum(current_max, x_ref[0, idx, :]),
        # If out of bounds, do nothing.
        lambda: current_max,
      )

    # Iterate over the kernel window to find the max values.
    final_max = lax.fori_loop(0, kernel_size, window_loop, max_vals)
    out_ref[0, j, :] = final_max

  # Use fori_loop for the main iteration over the output sequence.
  lax.fori_loop(0, out_ref.shape[1], loop_body, None)
