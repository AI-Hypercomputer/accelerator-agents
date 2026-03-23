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
padding = ((1, 1),)
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, input_length, in_channels))  # JAX uses channels-last convention
# The output length is calculated as:
# floor((input_length + padding_start + padding_end - kernel_size) / stride) + 1
# floor((128 + 1 + 1 - 4) / 2) + 1 = floor(126 / 2) + 1 = 63 + 1 = 64
output_length = 64


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for 1D average pooling.

  This kernel handles a single item from the batch. It pads the input,
  then iterates through the output positions, computing the average
  over a sliding window.

  Args:
    x_ref: A reference to the input data for a single batch item.
      Expected shape: (1, input_length, in_channels).
    out_ref: A reference to the output array for a single batch item,
      which will be written to in-place.
      Expected shape: (1, output_length, in_channels).
  """
  # Hard-coded parameters from the problem description
  kernel_size = 4
  stride = 2
  pad_start, pad_end = 1, 1

  # Get dimensions from the input reference
  # x_ref.shape is (1, 128, 32)
  in_channels = x_ref.shape[2]
  output_length = out_ref.shape[1]

  # Create a padded version of the input in SRAM.
  # The padding value for average pooling is 0.
  # The .at[...].set(...) pattern uses `scatter`, which is not supported on TPU.
  # We can construct the padded array using `concatenate` instead.
  zeros_start = jnp.zeros((pad_start, in_channels), dtype=x_ref.dtype)
  zeros_end = jnp.zeros((pad_end, in_channels), dtype=x_ref.dtype)
  # x_ref[0] has shape (input_length, in_channels)
  padded_x = jnp.concatenate([zeros_start, x_ref[0], zeros_end], axis=0)

  # Iterate over each position in the output's length dimension.
  for j in range(output_length):
    # Calculate the start index for the sliding window in the padded input.
    start_idx = j * stride

    # Extract the window of size `kernel_size` for all channels.
    # `dynamic_slice` is not supported on TPU, but standard array slicing is.
    window = padded_x[start_idx : start_idx + kernel_size, :]

    # Compute the mean over the window dimension (axis=0).
    # The result has shape (in_channels,).
    avg_value = jnp.mean(window, axis=0)

    # Write the computed average to the corresponding output slice.
    # out_ref has shape (1, output_length, in_channels).
    out_ref[0, j, :] = avg_value


# The kernel will be parallelized over the batch dimension.
# Each kernel instance will process one item from the batch.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, output_length, in_channels), x.dtype),
  # Grid is 1D, with one program per batch item.
  grid=(batch_size,),
  # Each program gets a slice of shape (1, input_length, in_channels)
  # corresponding to a single batch item.
  in_specs=[pl.BlockSpec(block_shape=(1, input_length, in_channels), index_map=lambda i: (i, 0, 0))],
  # Each program writes to a slice of shape (1, output_length, in_channels)
  # in the output array.
  out_specs=pl.BlockSpec(block_shape=(1, output_length, in_channels), index_map=lambda i: (i, 0, 0)),
)(x).block_until_ready()
