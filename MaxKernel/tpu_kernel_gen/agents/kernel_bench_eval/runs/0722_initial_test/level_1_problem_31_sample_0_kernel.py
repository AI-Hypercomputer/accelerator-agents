# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
alpha = 1.0
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


# Computation
def kernel(x_ref, alpha, out_ref):
  """Pallas kernel for the ELU activation function."""
  # Load the input block from SRAM into a register.
  x = x_ref[...]
  # Apply the ELU formula element-wise.
  # jnp.expm1(x) is equivalent to jnp.exp(x) - 1 but more numerically stable.
  result = jnp.where(x > 0, x, alpha * jnp.expm1(x))
  # Write the result to the output block in SRAM.
  out_ref[...] = result


result = pl.pallas_call(
  lambda x_ref, out_ref: kernel(x_ref, alpha, out_ref),
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // 16, dim // 128),
  in_specs=[pl.BlockSpec(block_shape=(16, 128), index_map=lambda i, j: (i * 16, j * 128))],
  out_specs=pl.BlockSpec(block_shape=(16, 128), index_map=lambda i, j: (i * 16, j * 128)),
)(x).block_until_ready()
