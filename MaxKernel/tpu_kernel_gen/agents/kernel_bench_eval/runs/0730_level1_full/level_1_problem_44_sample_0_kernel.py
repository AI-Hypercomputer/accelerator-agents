# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
input_length = 128
kernel_size = 4
stride = 2
padding = 1
key = random.PRNGKey(0)
# JAX/Pallas expect channels-last format: (batch, length, channels)
x = random.normal(key, (batch_size, input_length, in_channels))


def kernel(x_ref, out_ref):
  """Pallas kernel for 1D average pooling.

  This kernel handles one example from the batch. It iterates through
  the output positions, computing the average over the corresponding
  window by conditionally loading from the input to handle padding.
  """
  # Hard-coded constants from the problem description for clarity.
  input_length = 128
  kernel_size = 4
  stride = 2
  padding = 1
  in_channels = 32
  output_length = (input_length + 2 * padding - kernel_size) // stride + 1

  # Iterate over each position in the output's length dimension.
  for j in range(output_length):
    # Create an accumulator for the window sum for all channels.
    window_sum = jnp.zeros((in_channels,), dtype=x_ref.dtype)

    # Iterate over the kernel window.
    for k in range(kernel_size):
      # Calculate the corresponding index in the original, unpadded input.
      input_idx = j * stride - padding + k

      # Check if the index is within the valid bounds of the input array.
      # This conditional prevents out-of-bounds memory access.
      if (input_idx >= 0) & (input_idx < input_length):
        # If it's in bounds, load the data and add it to the accumulator.
        window_sum += x_ref[0, input_idx, :]

    # Compute the average over the kernel window.
    average_value = window_sum / kernel_size

    # Write the final averaged value to the output.
    out_ref[0, j, :] = average_value


# Computation
output_length = (input_length + 2 * padding - kernel_size) // stride + 1

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, output_length, in_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(1, input_length, in_channels), index_map=lambda i: (i, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, output_length, in_channels), index_map=lambda i: (i, 0, 0)),
)(x).block_until_ready()
