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


def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute the minimum along a specified axis.

  Args:
    x_ref: A reference to the input tensor block.
    out_ref: A reference to the output tensor block where the result is stored.
  """
  out_ref[...] = jnp.min(x_ref[...], axis=1)


# For TPU compatibility, the block size for the batch dimension needs to be divisible by 8.
# The batch size is 16, so we can use a block size of 8 or 16. Let's use 8 for more parallelism.
batch_block_size = 8

# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((x.shape[0], x.shape[2]), x.dtype),
  grid=(x.shape[0] // batch_block_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(batch_block_size, x.shape[1], x.shape[2]),
      index_map=lambda i: (i, 0, 0),
    )
  ],
  out_specs=pl.BlockSpec(block_shape=(batch_block_size, x.shape[2]), index_map=lambda i: (i, 0)),
)(x).block_until_ready()
