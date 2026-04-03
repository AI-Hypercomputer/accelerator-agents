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
block_batch = 8


# Computation
def kernel(x_ref, y_ref):
  """
  Pallas kernel to perform L2 normalization on a block of vectors.
  """
  # Load the input block from HBM into SRAM.
  x = x_ref[...]
  # Calculate the L2 norm for each row in the block.
  # The `axis=1` argument specifies that the norm should be computed along the rows (the `dim` dimension).
  # `keepdims=True` ensures that the resulting norm tensor has a shape of (block_batch, 1),
  # which allows it to be broadcast correctly for the division operation.
  norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
  # Perform the normalization by dividing the input block by the norms.
  # The result is written directly to the output reference, performing the computation in-place.
  y_ref[...] = x / norm


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // block_batch,),
  in_specs=[pl.BlockSpec(block_shape=(block_batch, dim), index_map=lambda i: (i, 0))],
  out_specs=pl.BlockSpec(block_shape=(block_batch, dim), index_map=lambda i: (i, 0)),
)(x).block_until_ready()
