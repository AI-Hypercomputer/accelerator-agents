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
# Define a block size for the batch dimension that adheres to TPU constraints.
# The block's second-to-last dimension (the batch dimension) must be divisible by 8.
batch_block_size = 8


# Computation
def kernel(x_ref, y_ref):
  """Pallas kernel for L2 normalization.

  This kernel normalizes each row of the input block `x_ref` by its L2 norm
  and writes the result to `y_ref`.

  Args:
    x_ref: A reference to the input block.
    y_ref: A reference to the output block.
  """
  # The input x_ref is a block of the original x matrix.
  # We compute the L2 norm for each row within this block.
  # axis=1 operates along the `dim` dimension of the (batch_block_size, dim) block.
  # keepdims=True makes the resulting norm broadcastable for the division.
  norm = jnp.linalg.norm(x_ref[...], ord=2, axis=1, keepdims=True)
  # We perform the normalization and write the result to the output reference.
  y_ref[...] = x_ref[...] / norm


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(shape=(batch_size, dim), dtype=jnp.float32),
  grid=(batch_size // batch_block_size,),
  in_specs=[pl.BlockSpec(block_shape=(batch_block_size, dim), index_map=lambda i: (i, 0))],
  out_specs=pl.BlockSpec(block_shape=(batch_block_size, dim), index_map=lambda i: (i, 0)),
)(x).block_until_ready()
