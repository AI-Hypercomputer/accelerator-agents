# Imports
import jax
import jax.random as random
from flax.linen import softplus
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise softplus."""
  # Load the input block from SRAM into registers.
  x = x_ref[...]
  # Compute the softplus function on the block.
  result = softplus(x)
  # Write the result to the corresponding output block in SRAM.
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[1] // block_dim,),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i: (0, i)),
)(x).block_until_ready()
