# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 32
depth = 64
height = 64
width = 64
kernel_size = 3
stride = 2
padding = 1
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, depth, height, width, channels))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for 3D average pooling.

  This kernel computes one slice of the output tensor of shape (1, 1, output_width, channels)
  for each grid element.

  Args:
    x_ref: Input tensor slice.
    out_ref: Output tensor slice to be written to.
  """
  # These constants are captured from the outer scope:
  # kernel_size, stride, padding, depth, height, width, channels

  # Get the grid indices for the current output slice.
  d_idx, h_idx = pl.program_id(1), pl.program_id(2)

  # The input `x_ref` is a slab of shape (1, kernel_size, kernel_size, width, channels).
  # First, we sum over the kernel's depth and height dimensions (axes 1 and 2).
  # This reduces the problem to a 1D pooling operation.
  sum_over_dh = jnp.sum(x_ref[...], axis=(1, 2))[0]

  # The `in_specs` handle padding for the depth and height dimensions implicitly.
  # We must manually pad the width dimension to handle pooling windows that
  # extend beyond the original tensor's boundaries.
  padded_sum_over_dh = jnp.pad(
    sum_over_dh, pad_width=((padding, padding), (0, 0)), mode="constant", constant_values=0.0
  )

  # To correctly calculate the average (with count_include_pad=False), we need
  # to find the number of valid (non-padded) elements in each window.
  # We first calculate the valid counts for the depth and height dimensions,
  # which are constant for the entire slice this kernel computes.
  d_start = d_idx * stride - padding
  h_start = h_idx * stride - padding
  valid_d = jnp.maximum(0, jnp.minimum(d_start + kernel_size, depth) - jnp.maximum(d_start, 0))
  valid_h = jnp.maximum(0, jnp.minimum(h_start + kernel_size, height) - jnp.maximum(h_start, 0))

  output_width = (width + 2 * padding - kernel_size) // stride + 1
  # Create an array to store the results of the loop.
  out_row_init = jnp.zeros((output_width, channels), dtype=x_ref.dtype)

  def body_fun(w_out, current_out_row):
    # Sum over the 1D window in the width dimension from the padded data.
    w_padded_start = w_out * stride
    # Use standard array slicing, which is supported, instead of dynamic_slice.
    window = jax.lax.slice(padded_sum_over_dh, (w_padded_start, 0), (w_padded_start + kernel_size, channels))
    total_sum = jnp.sum(window, axis=0)

    # Calculate the number of valid elements in the width dimension for this window.
    w_start = w_out * stride - padding
    valid_w = jnp.maximum(0, jnp.minimum(w_start + kernel_size, width) - jnp.maximum(w_start, 0))

    # The total divisor is the product of valid elements in all three dimensions.
    divisor = valid_d * valid_h * valid_w
    # Avoid division by zero for windows that are entirely in the padding area.
    divisor = jnp.maximum(1, divisor).astype(x_ref.dtype)

    # Update the output row with the computed average for the current window.
    # Use dynamic_update_slice, which is supported, instead of .at[...].set(...)
    return jax.lax.dynamic_update_slice(current_out_row, (total_sum / divisor)[None], (w_out, 0))

  # Iterate over the width dimension to compute the output row.
  out_row = jax.lax.fori_loop(0, output_width, body_fun, out_row_init)

  # Write the final computed row to the output reference.
  out_ref[...] = out_row.reshape(1, 1, 1, output_width, channels)


# Calculate output shape based on pooling parameters
output_depth = (depth + 2 * padding - kernel_size) // stride + 1
output_height = (height + 2 * padding - kernel_size) // stride + 1
output_width = (width + 2 * padding - kernel_size) // stride + 1

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, output_depth, output_height, output_width, channels), x.dtype),
  grid=(batch_size, output_depth, output_height),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, kernel_size, kernel_size, width, channels),
      index_map=lambda b, d, h: (b, d * stride - padding, h * stride - padding, 0, 0),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, 1, output_width, channels),
    index_map=lambda b, d, h: (b, d, h, 0, 0),
  ),
)(x).block_until_ready()
