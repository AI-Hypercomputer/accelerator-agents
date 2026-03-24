# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
length = 512

key = random.PRNGKey(0)
key_x, key_init = random.split(key)

# Note: JAX expects channels-last format (batch, length, channels)
x = random.normal(key_x, (batch_size, length, in_channels))

conv1d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size,),
  strides=1,
  padding="VALID",
  kernel_dilation=1,
  feature_group_count=1,
  use_bias=False,
)
params = conv1d.init(key_init, x)["params"]

# Calculate output shape from inputs
output_length = x.shape[1] - params["kernel"].shape[0] + 1
result_shape = (x.shape[0], output_length, params["kernel"].shape[2])

# Define block size for parallelization
block_len = 128


def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 1D convolution.

  This kernel computes a 1D convolution for a single block of the input data.
  The computation is parallelized across the batch and length dimensions as
  defined by the `grid` in the `pallas_call`.

  Args:
    x_ref: A reference to the input block. The block is larger than the output
      block to provide the necessary context for the convolution window.
    kernel_ref: A reference to the complete convolution kernel.
    out_ref: A reference to the output block to be written to.
  """
  # Get the index of the current block being processed along the length dimension.
  j_grid = pl.program_id(axis=1)

  # Get dimensions from the shapes of the references.
  kernel_len = kernel_ref.shape[0]
  in_channels = kernel_ref.shape[1]
  out_channels = kernel_ref.shape[2]

  # Load the kernel and the input block from memory into registers.
  x_block = x_ref[...]
  kernel_vals = kernel_ref[...]

  # Iterate over each position `k` in the output block.
  for k in range(out_ref.shape[1]):
    # Define the computation for when the index is in bounds.
    def true_fun(_):
      # Initialize an accumulator for the output pixel.
      acc = jnp.zeros(out_channels, dtype=x_ref.dtype)
      # Perform the convolution for the output at position `k`.
      for i in range(kernel_len):
        for c in range(in_channels):
          # The operation is a scaled sum, where each input `x_block[0, k + i, c]`
          # scales the corresponding kernel vector `kernel_vals[i, c, :]`.
          acc += x_block[0, k + i, c] * kernel_vals[i, c, :]
      # Write the accumulated result to the correct position in the output block.
      out_ref[0, k, :] = acc

    # Define the computation for when the index is out of bounds (do nothing).
    def false_fun(_):
      pass

    # Check if the global index is within the valid output bounds.
    is_in_bounds = j_grid * out_ref.shape[1] + k < output_length
    # Use lax.cond for conditional execution to avoid tracer errors.
    lax.cond(is_in_bounds, true_fun, false_fun, None)


# Computation
# The input block needs to be larger to account for the kernel window.
# input_block_len = block_len + kernel_size - 1 = 128 + 3 - 1 = 130
# For TPU compatibility, we round up to the nearest multiple of 8.
input_block_len = 136

result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(result_shape, x.dtype),
  grid=(x.shape[0], (output_length + block_len - 1) // block_len),
  in_specs=[
    pl.BlockSpec(block_shape=(1, input_block_len, x.shape[2]), index_map=lambda i, j: (i, j * block_len, 0)),
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i, j: (0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, block_len, params["kernel"].shape[2]), index_map=lambda i, j: (i, j * block_len, 0)
  ),
)(x, params["kernel"]).block_until_ready()
