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
# For element-wise operations, we can process the data in chunks.
# The grid will iterate over the input tensor.
# We must use a 2D block shape for TPUs, where the last two dimensions
# are divisible by 8 and 128 respectively.
block_m = 8
block_n = 2048


# Computation
def kernel(x_ref, y_ref):
  """Pallas kernel to compute tanh element-wise."""
  # Load the input block from HBM into SRAM, then into registers.
  x = x_ref[...]
  # Apply the tanh function.
  result = jnp.tanh(x)
  # Write the result to the output block in SRAM, which is then written back to HBM.
  y_ref[...] = result


y = pl.pallas_call(
  kernel,
  # The output shape is the same as the input shape.
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  # The grid iterates through each block of the tensor.
  grid=(x.shape[0] // block_m, x.shape[1] // block_n),
  # For each grid index (i, j), we select a block of the input.
  # The index_map scales the grid indices by the block shape to tile the input.
  in_specs=[pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n))],
  # The output block mapping is identical to the input.
  out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n)),
)(x).block_until_ready()
