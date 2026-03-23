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
# Flax uses channels-last convention: (batch, length, channels)
x = random.normal(key, (batch_size, input_length, in_channels))

# Calculate output shape
padded_length = input_length + padding[0][0] + padding[0][1]
output_length = (padded_length - kernel_size) // stride + 1
output_shape = (batch_size, output_length, in_channels)

# Pad the input tensor
x_padded = jnp.pad(x, ((0, 0),) + padding + ((0, 0),), "constant")


# Computation
def kernel(x_ref, o_ref):
  """Pallas kernel for 1D average pooling.

  This kernel computes the average pool for a single output element.
  The `pallas_call` is responsible for iterating over the grid and loading
  the correct input window into `x_ref` for each output element.

  Args:
    x_ref: A reference to the input block. Pallas loads a window of the
      input data into this reference. For an average pool, this window
      corresponds to the region over which the average is computed.
      The shape is (1, kernel_size, in_channels).
    o_ref: A reference to the output block. The kernel writes the
      computed average pool value here. The shape is (1, 1, in_channels).
  """
  # The input `x_ref` represents the window of data for the pooling operation,
  # automatically loaded by Pallas according to the `index_map` and `block_shape`
  # specified in the `pallas_call`.

  # We compute the sum over the window dimension (axis=1) and then divide by
  # the window size to get the average. This is more explicit than jnp.mean
  # and often preferred in Pallas kernels.
  sum_val = jnp.sum(x_ref, axis=1, keepdims=True)
  pooled_value = sum_val / kernel_size

  # The result is written to the output reference, performing the operation in-place.
  o_ref[...] = pooled_value


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, output_length),
  in_specs=pl.BlockSpec(
    (1, kernel_size, in_channels),
    lambda i, j: (i, j * stride, 0),
  ),
  out_specs=pl.BlockSpec(
    (1, 1, in_channels),
    lambda i, j: (i, j, 0),
  ),
)(x_padded).block_until_ready()
