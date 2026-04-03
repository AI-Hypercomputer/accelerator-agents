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
b_dim = 1024


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for hardtanh activation."""
  # Load the input block into registers.
  x = x_ref[...]
  # Apply the hardtanh function: max(-1, min(1, x)).
  # The computation is done element-wise on the block.
  result = jnp.maximum(-1.0, jnp.minimum(1.0, x))
  # Write the result to the output block.
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[1] // b_dim,),
  in_specs=[pl.BlockSpec((x.shape[0], b_dim), lambda i: (0, i * b_dim))],
  out_specs=[pl.BlockSpec((x.shape[0], b_dim), lambda i: (0, i * b_dim))],
)(x).block_until_ready()
