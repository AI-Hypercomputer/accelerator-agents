# Imports
import jax
import jax.nn
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_size = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for the SELU activation function."""
  # SELU constants
  alpha = 1.6732632423543772848170429916717
  lmbda = 1.0507009873554804934193349852946

  # Load the input block into SRAM
  x = x_ref[...]

  # Apply the SELU formula element-wise
  # selu(x) = lmbda * x if x > 0 else lmbda * (alpha * exp(x) - alpha)
  positive_values = lmbda * x
  negative_values = lmbda * (alpha * jnp.exp(x) - alpha)

  # Store the result in the output reference
  out_ref[...] = jnp.where(x > 0, positive_values, negative_values)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(dim // block_size,),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, block_size), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_size), index_map=lambda i: (0, i)),
)(x).block_until_ready()
