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


def kernel(x_ref, out_ref):
  """Pallas kernel to normalize vectors by the sum of their absolute values."""
  # This kernel now processes the entire matrix in a single launch.
  # We iterate through each row to perform the normalization.
  for i in range(x_ref.shape[0]):
    # Extract a single row.
    row = x_ref[i, :]
    # Calculate the sum of the absolute values of the row.
    denominator = jnp.sum(jnp.abs(row))
    # Divide the row by the calculated sum and write to the output.
    out_ref[i, :] = row / denominator


# Computation
# The grid is set to (1,) to launch a single kernel instance.
# The block_shape is set to the full shape of the input array `x`,
# which satisfies the TPU requirement that block dimensions must either be
# divisible by certain values or equal to the full array dimension.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1,),
  in_specs=[pl.BlockSpec(block_shape=x.shape, index_map=lambda i: (0, 0))],
  out_specs=pl.BlockSpec(block_shape=x.shape, index_map=lambda i: (0, 0)),
)(x).block_until_ready()
