# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 2000
dim = 2000
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


def kernel(x_ref, out_ref):
  """Pallas kernel for the GELU activation function."""
  # Load the input block into registers.
  x = x_ref[...]
  # Apply the GELU formula element-wise on the block.
  # This is a direct translation of the original JAX computation.
  result = 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))
  # Write the result to the output block.
  out_ref[...] = result


block_batch = 128
block_dim = 128
grid = (pl.cdiv(batch_size, block_batch), pl.cdiv(dim, block_dim))
spec = pl.BlockSpec(block_shape=(block_batch, block_dim), index_map=lambda i, j: (i, j))

# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=grid,
  in_specs=[spec],
  out_specs=spec,
)(x).block_until_ready()
