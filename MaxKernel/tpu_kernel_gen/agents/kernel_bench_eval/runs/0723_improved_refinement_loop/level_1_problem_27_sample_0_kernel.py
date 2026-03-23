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
block_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for the SELU activation function."""
  # Constants for SELU activation function, as defined in flax.linen
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946

  # Load the input data from SRAM into registers
  x = x_ref[...]

  # Apply the SELU formula element-wise
  # This is equivalent to: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
  result = jnp.where(x > 0, scale * x, scale * alpha * (jnp.exp(x) - 1.0))

  # Write the result back to the output buffer in SRAM
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[1] // block_dim,),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i: (0, i)),
)(x).block_until_ready()
