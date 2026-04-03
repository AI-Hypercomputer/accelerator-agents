# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
dim = 1
batch_size = 128
input_shape = (4000,)
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, *input_shape))
block_b = 8


# Computation
def kernel(x_ref, out_ref):
  """
  Computes the reverse cumulative sum along the last axis.
  """
  # Load the input block from HBM into SRAM.
  x = x_ref[...]
  # The computation is a reverse cumulative sum.
  # The "flip" method (x[:, ::-1]) is not supported on TPU.
  # An alternative implementation is:
  # total_sum - cumsum(x) + x
  total_sum = jnp.sum(x, axis=1, keepdims=True)
  forward_cumsum = jnp.cumsum(x, axis=1)
  result = total_sum - forward_cumsum + x
  # Write the result to the output buffer.
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // block_b,),
  in_specs=[pl.BlockSpec(block_shape=(block_b, input_shape[0]), index_map=lambda i: (i * block_b, 0))],
  out_specs=pl.BlockSpec(block_shape=(block_b, input_shape[0]), index_map=lambda i: (i * block_b, 0)),
)(x).block_until_ready()
