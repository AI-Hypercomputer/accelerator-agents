# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 2000
dim = 2000
block_b = 128
block_d = 128
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for the GELU activation function."""
  # Load the input block from SRAM into registers.
  x = x_ref[...]

  # Perform the GELU computation element-wise on the block.
  # This is a direct translation of the JAX source code.
  x_cubed = jnp.power(x, 3.0)
  inner_term = jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x_cubed)
  tanh_term = jnp.tanh(inner_term)

  result = 0.5 * x * (1.0 + tanh_term)

  # Write the computed block to the output buffer in SRAM.
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim), jnp.float32),
  grid=(
    (batch_size + block_b - 1) // block_b,
    (dim + block_d - 1) // block_d,
  ),
  in_specs=[pl.BlockSpec(block_shape=(block_b, block_d), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(block_b, block_d), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()
