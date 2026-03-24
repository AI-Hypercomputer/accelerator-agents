# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise SiLU activation."""
  # Load the input data from SRAM into registers.
  x = x_ref[...]
  # Apply the SiLU activation function.
  # The computation is done element-wise on the loaded block.
  result = x * jax.nn.sigmoid(x)
  # Write the result back to the output buffer in SRAM.
  out_ref[...] = result


# Define a smaller, manageable block size for the inner dimension
block_dim = 1024

# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size, dim // block_dim),
  in_specs=[pl.BlockSpec((1, block_dim), lambda i, j: (i, j * block_dim))],
  out_specs=[pl.BlockSpec((1, block_dim), lambda i, j: (i, j * block_dim))],
)(x).block_until_ready()
