# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_dim = 128


def kernel(x_ref, out_ref):
  # The computation is an element-wise GELU.
  # We load the input block, apply the function, and write to the output block.
  out_ref[...] = jax.nn.gelu(x_ref[...])


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(dim // block_dim,),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i)),
)(x).block_until_ready()
