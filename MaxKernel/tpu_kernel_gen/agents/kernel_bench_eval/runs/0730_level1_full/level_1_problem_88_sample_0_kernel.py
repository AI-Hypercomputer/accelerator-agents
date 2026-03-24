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
block_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for the GELU activation function."""
  # Load the input block into a variable.
  x = x_ref[...]

  # Apply the GELU activation formula.
  # The formula is: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  # This is an element-wise operation on the block.
  result = 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))

  # Write the computed block to the output reference.
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(
    (x.shape[0] + block_dim - 1) // block_dim,
    (x.shape[1] + block_dim - 1) // block_dim,
  ),
  in_specs=[pl.BlockSpec(block_shape=(block_dim, block_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(block_dim, block_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()
