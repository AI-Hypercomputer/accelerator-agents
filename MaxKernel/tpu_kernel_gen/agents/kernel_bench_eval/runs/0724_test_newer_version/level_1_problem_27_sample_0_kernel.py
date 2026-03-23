# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl


# Initialization
def kernel(x_ref, out_ref):
  """Pallas kernel for the SELU activation function."""
  # Constants for SELU activation
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946

  # Load the input data from SRAM into registers
  x = x_ref[...]

  # Apply the SELU function element-wise.
  # jnp.expm1(x) is not implemented in Pallas, so we use jnp.exp(x) - 1.
  result = scale * jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1.0))

  # Write the result to the output buffer in SRAM
  out_ref[...] = result


batch_size = 16
dim = 16384
# For 2D blocks on TPU, the last two dimensions of the block shape must be
# divisible by 8 and 128 respectively.
block_shape = (8, 128)
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))

# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim), jnp.float32),
  grid=(batch_size // block_shape[0], dim // block_shape[1]),
  in_specs=[
    pl.BlockSpec(
      block_shape=block_shape,
      index_map=lambda i, j: (i, j),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=block_shape,
    index_map=lambda i, j: (i, j),
  ),
)(x).block_until_ready()
