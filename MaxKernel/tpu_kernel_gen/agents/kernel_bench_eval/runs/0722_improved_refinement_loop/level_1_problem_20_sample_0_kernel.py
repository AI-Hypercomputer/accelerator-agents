# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
negative_slope = 0.01
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_dim = 2048


# Computation
def kernel(x_ref, out_ref, *, negative_slope: float):
  """Pallas kernel for Leaky ReLU activation."""
  # Load the input data from SRAM into registers.
  x = x_ref[...]
  # Apply the Leaky ReLU logic element-wise.
  # This is equivalent to nn.leaky_relu(x, negative_slope).
  result = jnp.where(x > 0, x, negative_slope * x)
  # Write the computed result back to the output buffer in SRAM.
  out_ref[...] = result


result = (
  pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    grid=(x.size // block_dim,),
    in_specs=[pl.BlockSpec(lambda i: (i,), (block_dim,))],
    out_specs=[pl.BlockSpec(lambda i: (i,), (block_dim,))],
    negative_slope=negative_slope,
  )(x.flatten())
  .reshape(x.shape)
  .block_until_ready()
)
