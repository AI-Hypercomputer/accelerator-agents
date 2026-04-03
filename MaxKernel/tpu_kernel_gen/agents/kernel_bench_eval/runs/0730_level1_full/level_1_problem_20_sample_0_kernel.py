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
block_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for leaky ReLU.

  Args:
    x_ref: Input block reference.
    out_ref: Output block reference.
  """
  # Load the input block into registers.
  x = x_ref[...]
  # Apply the leaky ReLU activation function element-wise.
  # The `negative_slope` variable is captured from the surrounding scope
  # where the kernel is defined and invoked.
  result = jnp.where(x > 0, x, x * negative_slope)
  # Write the result to the output block.
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[1] // block_dim,),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i: (0, i)),
)(x).block_until_ready()
