# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4000,)
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, *input_shape))


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute the exclusive cumulative sum along the last axis.
  """
  # The kernel processes a block of rows. We iterate through each one.
  for i in range(x_ref.shape[0]):
    # jax.lax.cumsum does not support the 'exclusive' argument.
    # We compute the inclusive sum first.
    inclusive_sum = jax.lax.cumsum(x_ref[i, :])

    # Then, we manually create the exclusive sum by shifting the inclusive
    # result and padding with a 0.
    scanned_row = jnp.zeros_like(x_ref[i, :])
    scanned_row = scanned_row.at[1:].set(inclusive_sum[:-1])

    # Write the entire resulting row to the output block.
    out_ref[i, :] = scanned_row


# The block shape for the second-to-last dimension must be divisible by 8 on TPU.
# We change the block shape from (1, 4000) to (8, 4000) and adjust the grid
# and index_map accordingly. The kernel is also updated to handle blocks of 8 rows.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // 8,),
  in_specs=[pl.BlockSpec(block_shape=(8, input_shape[0]), index_map=lambda i: (i * 8, 0))],
  out_specs=pl.BlockSpec(block_shape=(8, input_shape[0]), index_map=lambda i: (i * 8, 0)),
)(x).block_until_ready()
