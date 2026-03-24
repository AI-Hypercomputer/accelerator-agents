# Imports
import jax
import jax.nn
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_dim = 128


def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise ReLU."""
  # Load the input block from HBM into SRAM.
  x = x_ref[...]
  # Compute ReLU element-wise.
  result = jax.nn.relu(x)
  # Write the result to the output block in SRAM.
  out_ref[...] = result


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(dim // block_dim,),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i)),
)(x).block_until_ready()
