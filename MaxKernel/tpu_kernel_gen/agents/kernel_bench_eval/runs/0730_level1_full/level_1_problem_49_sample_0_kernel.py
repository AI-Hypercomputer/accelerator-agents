# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
dim = 1
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))
b_batch = 8


def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute the maximum value along axis 1.

  Args:
    x_ref: A reference to a block of the input tensor 'x'.
    out_ref: A reference to a block of the output tensor.
  """
  # The input block x_ref has shape (b_batch, dim1, dim2).
  # We compute the max along axis 1, resulting in a (b_batch, dim2) tensor.
  # This result is then written to the output block.
  out_ref[...] = jnp.max(x_ref[...], axis=1)


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(shape=(batch_size, dim2), dtype=x.dtype),
  grid=(batch_size // b_batch,),
  in_specs=[pl.BlockSpec(block_shape=(b_batch, dim1, dim2), index_map=lambda i: (i, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(b_batch, dim2), index_map=lambda i: (i, 0)),
)(x).block_until_ready()
