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
block_d = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for SeLU activation."""
  # SeLU constants
  alpha = 1.6732632423543772
  scale = 1.0507009873554805

  # Load data from HBM to SRAM
  x = x_ref[...]

  # Perform the SeLU computation
  # Using jnp.exp(x) - 1 which is equivalent to jnp.expm1(x)
  result = scale * jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))

  # Write the result back to HBM
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(dim // block_d,),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, block_d), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_d), index_map=lambda i: (0, i)),
)(x).block_until_ready()
