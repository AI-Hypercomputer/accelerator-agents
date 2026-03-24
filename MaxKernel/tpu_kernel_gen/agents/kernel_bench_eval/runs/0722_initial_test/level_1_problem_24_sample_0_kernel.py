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


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for log_softmax."""
  # Load the input array.
  x = x_ref[...]

  # Compute log_softmax for each row.
  # This uses the standard numerically stable implementation:
  # log_softmax(x) = x - log(sum(exp(x)))
  # Let m = max(x)
  # log_softmax(x) = x - m - log(sum(exp(x - m)))
  max_val = jnp.max(x, axis=1, keepdims=True)
  x_minus_max = x - max_val
  exp_x = jnp.exp(x_minus_max)
  sum_exp_x = jnp.sum(exp_x, axis=1, keepdims=True)
  log_sum_exp_x = jnp.log(sum_exp_x)

  # Write the result to the output reference.
  out_ref[...] = x_minus_max - log_sum_exp_x


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1,),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, dim), index_map=lambda i: (0, 0))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, dim), index_map=lambda i: (0, 0)),
)(x).block_until_ready()
