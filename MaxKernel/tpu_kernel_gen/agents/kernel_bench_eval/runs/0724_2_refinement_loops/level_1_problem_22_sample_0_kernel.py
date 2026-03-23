# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel for element-wise tanh.

  Args:
    x_ref: A reference to a block of the input array.
    out_ref: A reference to a block of the output array to write results to.
  """
  # Load the input data for the current block from SRAM into registers.
  x = x_ref[...]
  # Apply the tanh function element-wise.
  result = jnp.tanh(x)
  # Write the computed result back to the corresponding output block in SRAM.
  out_ref[...] = result


# We choose a block size for the dimension we are parallelizing over.
# For TPU compatibility, this dimension must be divisible by 128.
block_dim = 128

# The grid will have `dim // block_dim` instances.
grid_size = dim // block_dim

result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim), jnp.float32),
  # Create a 1D grid to iterate over the chunks of the `dim` dimension.
  grid=(grid_size,),
  # Each kernel instance receives a (batch_size, block_dim) slice of the input.
  in_specs=[pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i))],
  # The output specification mirrors the input specification.
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i)),
)(x).block_until_ready()
